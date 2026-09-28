from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil


GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(slots=True, frozen=True)
class MemoryPolicy:
    total_bytes: int
    reserve_system_bytes: int
    max_worker_rss_bytes: int
    max_worker_address_space_bytes: int
    max_swap_growth_bytes: int

    @classmethod
    def auto(cls) -> "MemoryPolicy":
        total = int(psutil.virtual_memory().total)
        reserve = int(max(12 * GIB, total * 0.25))
        # Keep one parser worker bounded so a pathological source file cannot
        # consume the workstation. On a ~64 GiB host this yields ~8 GiB RSS and
        # 10 GiB address-space hard cap, while ~16 GiB remains reserved for OS/UI.
        worker_rss = int(min(8 * GIB, max(4 * GIB, total * 0.14)))
        worker_as = int(min(10 * GIB, max(5 * GIB, total * 0.18)))
        return cls(
            total_bytes=total,
            reserve_system_bytes=reserve,
            max_worker_rss_bytes=worker_rss,
            max_worker_address_space_bytes=worker_as,
            max_swap_growth_bytes=512 * MIB,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "reserve_system_bytes": self.reserve_system_bytes,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "max_worker_address_space_bytes": self.max_worker_address_space_bytes,
            "max_swap_growth_bytes": self.max_swap_growth_bytes,
            "reserve_system_gb": round(self.reserve_system_bytes / GIB, 2),
            "max_worker_rss_gb": round(self.max_worker_rss_bytes / GIB, 2),
            "max_worker_address_space_gb": round(
                self.max_worker_address_space_bytes / GIB, 2
            ),
            "max_swap_growth_gb": round(self.max_swap_growth_bytes / GIB, 2),
        }


def apply_linux_address_space_limit(limit_bytes: int) -> bool:
    """Hard-cap parser worker virtual memory before opening source content."""
    if sys.platform != "linux":
        return False
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = int(limit_bytes)
        if hard != resource.RLIM_INFINITY:
            target = min(target, int(hard))
        resource.setrlimit(resource.RLIMIT_AS, (target, target))
        return True
    except Exception:
        return False


def process_tree_rss(pid: int) -> int:
    try:
        proc = psutil.Process(pid)
        total = int(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


class HostMemoryWatchdog:
    """Fail-fast host guard: protecting the OS is more important than a run."""

    def __init__(
        self,
        *,
        reserve_bytes: int,
        log_dir: str | Path,
        interval_seconds: float = 0.25,
        on_warning: Callable[[str], None] | None = None,
        max_swap_growth_bytes: int = 512 * MIB,
    ):
        self.reserve_bytes = int(reserve_bytes)
        self.log_dir = Path(log_dir)
        self.interval_seconds = float(interval_seconds)
        self.on_warning = on_warning
        self.max_swap_growth_bytes = int(max_swap_growth_bytes)
        self._swap_baseline = int(psutil.swap_memory().used)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="oracle-lite-host-memory-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            available = int(psutil.virtual_memory().available)
            swap_used = int(psutil.swap_memory().used)
            swap_growth = max(0, swap_used - self._swap_baseline)

            ram_critical = available < self.reserve_bytes
            swap_critical = swap_growth > self.max_swap_growth_bytes
            if not ram_critical and not swap_critical:
                continue

            if ram_critical:
                reason = (
                    "available host RAM "
                    f"{available / GIB:.2f} GiB fell below reserved "
                    f"{self.reserve_bytes / GIB:.2f} GiB"
                )
            else:
                reason = (
                    "swap grew by "
                    f"{swap_growth / GIB:.2f} GiB above startup baseline "
                    f"(limit {self.max_swap_growth_bytes / GIB:.2f} GiB)"
                )

            message = (
                "EMERGENCY MEMORY STOP: "
                + reason
                + ". Oracle-Lite is terminating immediately to protect the OS."
            )
            try:
                marker = self.log_dir / "memory-emergency.log"
                with marker.open("a", encoding="utf-8") as f:
                    f.write(f"{time.time():.3f} {message}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass

            if self.on_warning is not None:
                try:
                    self.on_warning(message)
                except Exception:
                    pass

            # Immediate termination is intentional. Waiting for normal unwinding
            # under memory pressure risks swapping the whole workstation.
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            finally:
                os._exit(75)

from __future__ import annotations

import gc
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
    resume_system_bytes: int
    max_worker_rss_bytes: int
    max_worker_address_space_bytes: int
    max_swap_growth_bytes: int

    @classmethod
    def auto(cls) -> "MemoryPolicy":
        total = int(psutil.virtual_memory().total)

        # Scale down sanely on small CI/dev hosts while staying deliberately
        # conservative on the target ~64 GiB Ubuntu workstation.
        if total >= 48 * GIB:
            reserve_target = max(12 * GIB, total * 0.25)
        else:
            reserve_target = max(2 * GIB, total * 0.30)
        reserve = int(min(reserve_target, total * 0.50))
        resume = int(min(total * 0.60, reserve + max(2 * GIB, total * 0.06)))

        # On ~64 GiB: ~8 GiB RSS cap and ~12 GiB address-space hard cap.
        worker_rss = int(min(8 * GIB, max(2 * GIB, total * 0.125), total * 0.25))
        worker_as = int(min(12 * GIB, max(3 * GIB, total * 0.19), total * 0.35))
        return cls(
            total_bytes=total,
            reserve_system_bytes=reserve,
            resume_system_bytes=max(resume, reserve),
            max_worker_rss_bytes=worker_rss,
            max_worker_address_space_bytes=worker_as,
            max_swap_growth_bytes=512 * MIB,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "reserve_system_bytes": self.reserve_system_bytes,
            "resume_system_bytes": self.resume_system_bytes,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "max_worker_address_space_bytes": self.max_worker_address_space_bytes,
            "max_swap_growth_bytes": self.max_swap_growth_bytes,
            "reserve_system_gb": round(self.reserve_system_bytes / GIB, 2),
            "resume_system_gb": round(self.resume_system_bytes / GIB, 2),
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

        _, hard = resource.getrlimit(resource.RLIMIT_AS)
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


def wait_for_safe_memory(
    policy: MemoryPolicy,
    *,
    on_wait: Callable[[dict], None] | None = None,
    poll_seconds: float = 1.0,
    stable_samples: int = 3,
) -> None:
    """Pause the pipeline until RAM is safely above the resume threshold.

    Memory pressure is backpressure, not failure. Hysteresis prevents rapid
    pause/resume oscillation around one threshold.
    """
    stable = 0
    while stable < stable_samples:
        gc.collect()
        vm = psutil.virtual_memory()
        available = int(vm.available)
        safe = available >= policy.resume_system_bytes

        if on_wait is not None:
            on_wait({
                "available_bytes": available,
                "available_gb": round(available / GIB, 2),
                "reserve_gb": round(policy.reserve_system_bytes / GIB, 2),
                "resume_gb": round(policy.resume_system_bytes / GIB, 2),
                "safe": safe,
            })

        if safe:
            stable += 1
        else:
            stable = 0
        if stable < stable_samples:
            time.sleep(poll_seconds)


class HostMemoryWatchdog:
    """Non-destructive pressure observer.

    This class no longer terminates Oracle-Lite. The actual pipeline uses
    wait_for_safe_memory() and worker restarts to apply backpressure.
    """

    def __init__(
        self,
        *,
        reserve_bytes: int,
        log_dir: str | Path,
        interval_seconds: float = 0.5,
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
        self._in_pressure = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
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
            pressure = (
                available < self.reserve_bytes
                or swap_growth > self.max_swap_growth_bytes
            )

            if pressure and not self._in_pressure and self.on_warning is not None:
                self.on_warning(
                    "MEMORY PRESSURE: Oracle-Lite will pause/restart the current "
                    "memory-heavy stage instead of exhausting host RAM."
                )
            self._in_pressure = pressure

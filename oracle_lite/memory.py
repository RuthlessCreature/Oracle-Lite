from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
    cgroup_memory_max_bytes: int
    cgroup_swap_max_bytes: int

    @classmethod
    def auto(cls) -> "MemoryPolicy":
        total = int(psutil.virtual_memory().total)
        desired_reserve = int(max(12 * GIB, total * 0.25))
        # Never let a fixed minimum reserve consume the whole machine on smaller
        # hosts. Always leave at least 1 GiB available to the Oracle-Lite cgroup.
        reserve = min(desired_reserve, max(0, total - GIB))
        cgroup_memory_max = max(GIB, total - reserve)

        # Derive parser caps from the already-safe cgroup budget. On a ~64 GiB
        # workstation these remain ~8 GiB RSS and ~10 GiB address space.
        worker_rss = int(min(8 * GIB, max(GIB, cgroup_memory_max * 0.50)))
        worker_as = int(min(10 * GIB, max(2 * GIB, cgroup_memory_max * 0.75)))
        worker_rss = min(worker_rss, cgroup_memory_max)
        worker_as = min(worker_as, cgroup_memory_max)

        return cls(
            total_bytes=total,
            reserve_system_bytes=reserve,
            max_worker_rss_bytes=worker_rss,
            max_worker_address_space_bytes=worker_as,
            max_swap_growth_bytes=512 * MIB,
            cgroup_memory_max_bytes=cgroup_memory_max,
            cgroup_swap_max_bytes=512 * MIB,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "reserve_system_bytes": self.reserve_system_bytes,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "max_worker_address_space_bytes": self.max_worker_address_space_bytes,
            "max_swap_growth_bytes": self.max_swap_growth_bytes,
            "cgroup_memory_max_bytes": self.cgroup_memory_max_bytes,
            "cgroup_swap_max_bytes": self.cgroup_swap_max_bytes,
            "reserve_system_gb": round(self.reserve_system_bytes / GIB, 2),
            "max_worker_rss_gb": round(self.max_worker_rss_bytes / GIB, 2),
            "max_worker_address_space_gb": round(
                self.max_worker_address_space_bytes / GIB, 2
            ),
            "max_swap_growth_gb": round(self.max_swap_growth_bytes / GIB, 2),
            "cgroup_memory_max_gb": round(self.cgroup_memory_max_bytes / GIB, 2),
            "cgroup_swap_max_gb": round(self.cgroup_swap_max_bytes / GIB, 2),
        }



MEMORY_SCOPE_ENV = "ORACLE_LITE_MEMORY_SCOPE"


def build_systemd_memory_scope_command(
    policy: MemoryPolicy,
    argv: Sequence[str] | None = None,
) -> list[str]:
    """Build the fail-closed systemd scope command used on Linux/Ubuntu."""
    systemd_run = shutil.which("systemd-run")
    env_bin = shutil.which("env")
    if not systemd_run:
        raise RuntimeError(
            "systemd-run is required for Oracle-Lite Linux hard memory fence."
        )
    if not env_bin:
        raise RuntimeError("env executable is required for Oracle-Lite memory scope.")

    forwarded = list(sys.argv[1:] if argv is None else argv)
    return [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={policy.cgroup_memory_max_bytes}",
        "-p",
        f"MemorySwapMax={policy.cgroup_swap_max_bytes}",
        env_bin,
        f"{MEMORY_SCOPE_ENV}=1",
        sys.executable,
        "-m",
        "oracle_lite.cli",
        *forwarded,
    ]


def ensure_linux_memory_scope(
    policy: MemoryPolicy,
    argv: Sequence[str] | None = None,
) -> bool:
    """Re-exec Oracle-Lite inside a user systemd/cgroup hard memory fence."""
    if sys.platform != "linux":
        return False
    if os.environ.get(MEMORY_SCOPE_ENV) == "1":
        return False

    systemd_run = shutil.which("systemd-run")
    if not systemd_run:
        raise RuntimeError(
            "Oracle-Lite refuses to run memory-heavy work on Linux without "
            "systemd-run/cgroup memory isolation."
        )

    probe = subprocess.run(
        [systemd_run, "--user", "--scope", "--quiet", "true"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip()
        raise RuntimeError(
            "Oracle-Lite could not establish the required user cgroup memory "
            f"scope and will not run unguarded. systemd-run: {detail or probe.returncode}"
        )

    command = build_systemd_memory_scope_command(policy, argv=argv)
    os.execv(command[0], command)
    return True

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

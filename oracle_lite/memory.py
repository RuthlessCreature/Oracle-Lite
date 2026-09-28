from __future__ import annotations

import os
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
MEMORY_SCOPE_ENV = "ORACLE_LITE_MEMORY_SCOPE"


@dataclass(slots=True, frozen=True)
class MemoryPolicy:
    total_bytes: int
    reserve_system_bytes: int
    resume_system_bytes: int
    max_worker_rss_bytes: int
    max_worker_address_space_bytes: int
    max_swap_growth_bytes: int
    cgroup_memory_max_bytes: int
    cgroup_swap_max_bytes: int

    @classmethod
    def auto(cls) -> "MemoryPolicy":
        total = int(psutil.virtual_memory().total)
        desired_reserve = int(max(12 * GIB, total * 0.25))
        reserve = min(desired_reserve, max(0, total - GIB))
        hysteresis = int(max(2 * GIB, total * 0.05))
        resume = min(total, reserve + hysteresis)

        # Leave headroom below the kernel hard fence. Normal flow control pauses
        # before this budget is exhausted; the cgroup remains the last defense.
        cgroup_budget = max(GIB, total - reserve - min(4 * GIB, hysteresis))

        worker_rss = int(min(8 * GIB, max(GIB, cgroup_budget * 0.50)))
        worker_as = int(min(10 * GIB, max(2 * GIB, cgroup_budget * 0.75)))
        worker_rss = min(worker_rss, cgroup_budget)
        worker_as = min(worker_as, cgroup_budget)

        return cls(
            total_bytes=total,
            reserve_system_bytes=reserve,
            resume_system_bytes=resume,
            max_worker_rss_bytes=worker_rss,
            max_worker_address_space_bytes=worker_as,
            max_swap_growth_bytes=512 * MIB,
            cgroup_memory_max_bytes=cgroup_budget,
            cgroup_swap_max_bytes=512 * MIB,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "reserve_system_bytes": self.reserve_system_bytes,
            "resume_system_bytes": self.resume_system_bytes,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "max_worker_address_space_bytes": self.max_worker_address_space_bytes,
            "max_swap_growth_bytes": self.max_swap_growth_bytes,
            "cgroup_memory_max_bytes": self.cgroup_memory_max_bytes,
            "cgroup_swap_max_bytes": self.cgroup_swap_max_bytes,
            "reserve_system_gb": round(self.reserve_system_bytes / GIB, 2),
            "resume_system_gb": round(self.resume_system_bytes / GIB, 2),
            "max_worker_rss_gb": round(self.max_worker_rss_bytes / GIB, 2),
            "max_worker_address_space_gb": round(
                self.max_worker_address_space_bytes / GIB, 2
            ),
            "max_swap_growth_gb": round(self.max_swap_growth_bytes / GIB, 2),
            "cgroup_memory_max_gb": round(self.cgroup_memory_max_bytes / GIB, 2),
            "cgroup_swap_max_gb": round(self.cgroup_swap_max_bytes / GIB, 2),
        }


def build_systemd_memory_scope_command(
    policy: MemoryPolicy,
    argv: Sequence[str] | None = None,
) -> list[str]:
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
    """Re-exec memory-heavy work inside a fail-closed user cgroup."""
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
    if sys.platform != "linux":
        return False
    try:
        import resource

        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
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


class MemoryPressureGate:
    """Pause/resume flow control for host RAM pressure.

    Memory pressure is not a failure. The gate enters PAUSED_MEMORY, blocks new
    allocations at safe points, and resumes only after hysteresis is satisfied.
    The Linux cgroup remains an independent last-resort hard fence.
    """

    def __init__(
        self,
        policy: MemoryPolicy,
        *,
        log_dir: str | Path,
        interval_seconds: float = 0.25,
        on_pause: Callable[[str], None] | None = None,
        on_resume: Callable[[str], None] | None = None,
    ):
        self.policy = policy
        self.log_dir = Path(log_dir)
        self.interval_seconds = float(interval_seconds)
        self.on_pause = on_pause
        self.on_resume = on_resume
        self._swap_baseline = int(psutil.swap_memory().used)
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._pause_until = 0.0
        self._reason: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def reason(self) -> str | None:
        with self._condition:
            return self._reason

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="oracle-lite-memory-pressure-gate",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def request_pause(self, reason: str, *, minimum_seconds: float = 2.0) -> None:
        with self._condition:
            self._pause_until = max(
                self._pause_until,
                time.monotonic() + max(0.0, minimum_seconds),
            )
            first = not self._paused.is_set()
            self._reason = reason
            self._paused.set()
            self._condition.notify_all()
        if first:
            self._record("PAUSED_MEMORY", reason)
            if self.on_pause is not None:
                self.on_pause(reason)

    def wait_until_safe(
        self,
        *,
        context: str = "pipeline",
        poll_seconds: float = 0.5,
    ) -> float:
        """Block at a safe point until RAM pressure has cleared."""
        started: float | None = None
        while not self._stop.is_set():
            if not self.paused and self._currently_safe_for_new_work():
                return 0.0 if started is None else time.monotonic() - started

            if started is None:
                started = time.monotonic()
            if not self.paused:
                self.request_pause(
                    f"Waiting for memory before {context}",
                    minimum_seconds=1.0,
                )
            with self._condition:
                self._condition.wait(timeout=poll_seconds)

        return 0.0 if started is None else time.monotonic() - started

    def _currently_safe_for_new_work(self) -> bool:
        vm = psutil.virtual_memory()
        return int(vm.available) >= self.policy.reserve_system_bytes

    def _record(self, state: str, message: str) -> None:
        try:
            path = self.log_dir / "memory-pressure.log"
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{time.time():.3f} {state} {message}\n")
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            vm = psutil.virtual_memory()
            available = int(vm.available)
            swap_used = int(psutil.swap_memory().used)
            swap_growth = max(0, swap_used - self._swap_baseline)

            pressure = (
                available < self.policy.reserve_system_bytes
                or swap_growth > self.policy.max_swap_growth_bytes
            )

            if pressure and not self.paused:
                if available < self.policy.reserve_system_bytes:
                    reason = (
                        f"Available RAM {available / GIB:.2f} GiB is below "
                        f"reserve {self.policy.reserve_system_bytes / GIB:.2f} GiB"
                    )
                else:
                    reason = (
                        f"Swap grew {swap_growth / GIB:.2f} GiB above baseline "
                        f"(limit {self.policy.max_swap_growth_bytes / GIB:.2f} GiB)"
                    )
                self.request_pause(reason, minimum_seconds=2.0)
                continue

            if not self.paused:
                continue

            # Hysteresis: do not bounce between pause/run around one threshold.
            can_resume = (
                time.monotonic() >= self._pause_until
                and available >= self.policy.resume_system_bytes
            )
            if not can_resume:
                continue

            with self._condition:
                reason = self._reason or "memory pressure cleared"
                self._reason = None
                self._paused.clear()
                self._swap_baseline = swap_used
                self._condition.notify_all()

            message = (
                f"Memory recovered to {available / GIB:.2f} GiB available; "
                "pipeline may resume"
            )
            self._record("RESUMED", message)
            if self.on_resume is not None:
                self.on_resume(message)


# Compatibility alias for code/tests outside the normal CLI path.
HostMemoryWatchdog = MemoryPressureGate

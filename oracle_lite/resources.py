from __future__ import annotations

import gc
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil

GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(slots=True, frozen=True)
class HostResourcePolicy:
    ram_total_bytes: int
    ram_reserve_bytes: int
    ram_resume_bytes: int
    parser_address_space_bytes: int
    parser_rss_budget_bytes: int
    disk_total_bytes: int
    disk_reserve_bytes: int
    disk_resume_bytes: int
    gpu_step_reserve_bytes: int
    model_download_budget_bytes: int
    training_write_budget_bytes: int

    @classmethod
    def auto(cls, output_dir: str | Path) -> "HostResourcePolicy":
        ram_total = int(psutil.virtual_memory().total)

        # On the target ~64 GiB workstation:
        # - parser may start only around >=26 GiB available
        # - one parser can address at most ~6 GiB
        # - therefore Oracle-Lite itself cannot consume the OS reserve.
        if ram_total >= 48 * GIB:
            ram_reserve = int(max(16 * GIB, ram_total * 0.30))
            parser_as = int(min(6 * GIB, ram_total * 0.10))
            parser_rss = int(min(5 * GIB, ram_total * 0.08))
        else:
            ram_reserve = int(min(ram_total * 0.45, max(2 * GIB, ram_total * 0.30)))
            parser_as = int(min(4 * GIB, max(2 * GIB, ram_total * 0.15)))
            parser_rss = int(min(3 * GIB, max(1 * GIB, ram_total * 0.12)))

        ram_resume = int(
            min(
                ram_total * 0.70,
                ram_reserve + parser_as + max(1 * GIB, ram_total * 0.02),
            )
        )
        ram_resume = max(ram_resume, ram_reserve)

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(output)
        disk_total = int(disk.total)
        if disk_total >= 500 * GIB:
            disk_reserve = int(max(80 * GIB, disk_total * 0.10))
        else:
            disk_reserve = int(max(15 * GIB, disk_total * 0.12))
        disk_reserve = int(min(disk_reserve, disk_total * 0.25))
        disk_resume = int(
            min(
                disk_total * 0.40,
                disk_reserve + max(10 * GIB, disk_total * 0.02),
            )
        )
        disk_resume = max(disk_resume, disk_reserve)

        return cls(
            ram_total_bytes=ram_total,
            ram_reserve_bytes=ram_reserve,
            ram_resume_bytes=ram_resume,
            parser_address_space_bytes=parser_as,
            parser_rss_budget_bytes=parser_rss,
            disk_total_bytes=disk_total,
            disk_reserve_bytes=disk_reserve,
            disk_resume_bytes=disk_resume,
            gpu_step_reserve_bytes=int(1.5 * GIB),
            model_download_budget_bytes=30 * GIB,
            training_write_budget_bytes=10 * GIB,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ram_total_gb": round(self.ram_total_bytes / GIB, 2),
            "ram_reserve_gb": round(self.ram_reserve_bytes / GIB, 2),
            "ram_resume_gb": round(self.ram_resume_bytes / GIB, 2),
            "parser_as_cap_gb": round(self.parser_address_space_bytes / GIB, 2),
            "parser_rss_budget_gb": round(self.parser_rss_budget_bytes / GIB, 2),
            "disk_total_gb": round(self.disk_total_bytes / GIB, 2),
            "disk_reserve_gb": round(self.disk_reserve_bytes / GIB, 2),
            "disk_resume_gb": round(self.disk_resume_bytes / GIB, 2),
            "gpu_step_reserve_gb": round(self.gpu_step_reserve_bytes / GIB, 2),
            "model_download_budget_gb": round(self.model_download_budget_bytes / GIB, 2),
            "training_write_budget_gb": round(self.training_write_budget_bytes / GIB, 2),
        }


def wait_for_ram(
    policy: HostResourcePolicy,
    *,
    extra_required_bytes: int = 0,
    on_wait: Callable[[dict], None] | None = None,
    poll_seconds: float = 1.0,
    stable_samples: int = 3,
) -> None:
    target = min(
        policy.ram_total_bytes,
        policy.ram_resume_bytes + max(0, int(extra_required_bytes)),
    )
    stable = 0
    while stable < stable_samples:
        gc.collect()
        vm = psutil.virtual_memory()
        available = int(vm.available)
        safe = available >= target
        if on_wait is not None:
            on_wait({
                "resource": "ram",
                "safe": safe,
                "available_bytes": available,
                "available_gb": round(available / GIB, 2),
                "target_bytes": target,
                "target_gb": round(target / GIB, 2),
                "reserve_gb": round(policy.ram_reserve_bytes / GIB, 2),
            })
        stable = stable + 1 if safe else 0
        if stable < stable_samples:
            time.sleep(poll_seconds)


def wait_for_disk(
    output_dir: str | Path,
    policy: HostResourcePolicy,
    *,
    required_bytes: int = 0,
    on_wait: Callable[[dict], None] | None = None,
    poll_seconds: float = 2.0,
    stable_samples: int = 2,
) -> None:
    output = Path(output_dir).expanduser().resolve()
    required = max(0, int(required_bytes))
    stable = 0
    while stable < stable_samples:
        usage = shutil.disk_usage(output)
        target = min(
            int(usage.total),
            policy.disk_resume_bytes + required,
        )
        free = int(usage.free)
        safe = free >= target
        if on_wait is not None:
            on_wait({
                "resource": "disk",
                "safe": safe,
                "free_bytes": free,
                "free_gb": round(free / GIB, 2),
                "target_bytes": target,
                "target_gb": round(target / GIB, 2),
                "reserve_gb": round(policy.disk_reserve_bytes / GIB, 2),
                "required_gb": round(required / GIB, 2),
            })
        stable = stable + 1 if safe else 0
        if stable < stable_samples:
            time.sleep(poll_seconds)


def current_gpu_memory(index: int = 0) -> dict | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "total_bytes": int(mem.total),
                "used_bytes": int(mem.used),
                "free_bytes": int(mem.free),
                "total_gb": round(mem.total / GIB, 2),
                "used_gb": round(mem.used / GIB, 2),
                "free_gb": round(mem.free / GIB, 2),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def wait_for_vram(
    *,
    required_free_bytes: int,
    on_wait: Callable[[dict], None] | None = None,
    poll_seconds: float = 2.0,
    stable_samples: int = 2,
    index: int = 0,
) -> None:
    required = max(0, int(required_free_bytes))
    stable = 0
    while stable < stable_samples:
        state = current_gpu_memory(index)
        # If NVML is unavailable, let the caller's torch.cuda.mem_get_info()
        # perform the authoritative check instead of inventing a value here.
        if state is None:
            return
        safe = int(state["free_bytes"]) >= required
        payload = {
            "resource": "vram",
            "safe": safe,
            **state,
            "required_free_bytes": required,
            "required_free_gb": round(required / GIB, 2),
        }
        if on_wait is not None:
            on_wait(payload)
        stable = stable + 1 if safe else 0
        if stable < stable_samples:
            time.sleep(poll_seconds)


def recommended_pdf_max_pixels(memory_level: int) -> int:
    # Explicit pixel-area ceilings bound pixmap RAM and later visual-token count.
    return {
        0: 1_048_576,
        1: 786_432,
        2: 524_288,
        3: 393_216,
        4: 262_144,
    }.get(max(0, int(memory_level)), 262_144)


def scale_for_pixel_budget(width: float, height: float, max_pixels: int) -> float:
    area = max(1.0, float(width) * float(height))
    return min(1.0, math.sqrt(max(1, int(max_pixels)) / area))

from __future__ import annotations

from dataclasses import dataclass

import psutil


GIB = 1024 ** 3


@dataclass(slots=True, frozen=True)
class MemoryPolicy:
    total_bytes: int
    reserve_system_bytes: int
    max_worker_rss_bytes: int

    @classmethod
    def auto(cls) -> "MemoryPolicy":
        total = int(psutil.virtual_memory().total)
        reserve = int(max(6 * GIB, total * 0.20))
        # Keep one parser worker bounded so a pathological source file cannot
        # consume the whole workstation. The cap is intentionally internal and
        # not another user-facing config field.
        worker_cap = int(min(10 * GIB, max(4 * GIB, total * 0.18)))
        return cls(
            total_bytes=total,
            reserve_system_bytes=reserve,
            max_worker_rss_bytes=worker_cap,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "reserve_system_bytes": self.reserve_system_bytes,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "reserve_system_gb": round(self.reserve_system_bytes / GIB, 2),
            "max_worker_rss_gb": round(self.max_worker_rss_bytes / GIB, 2),
        }


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

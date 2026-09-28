import subprocess
import sys

import pytest

from oracle_lite.memory import GIB, MIB, MemoryPolicy, wait_for_safe_memory


def test_auto_memory_policy_keeps_conservative_system_reserve_and_worker_cap():
    policy = MemoryPolicy.auto()
    assert policy.total_bytes > 0
    assert policy.reserve_system_bytes >= 12 * GIB
    assert policy.reserve_system_bytes < policy.total_bytes
    assert 4 * GIB <= policy.max_worker_rss_bytes <= 8 * GIB
    assert 6 * GIB <= policy.max_worker_address_space_bytes <= 12 * GIB
    assert policy.resume_system_bytes >= policy.reserve_system_bytes
    assert policy.max_worker_rss_bytes < policy.total_bytes
    assert policy.max_worker_address_space_bytes < policy.total_bytes
    assert policy.max_swap_growth_bytes == 512 * MIB


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is a Linux safety guard")
def test_linux_address_space_limit_blocks_oversized_allocation():
    code = r"""
from oracle_lite.memory import apply_linux_address_space_limit, MIB
ok = apply_linux_address_space_limit(768 * MIB)
print("limit", ok, flush=True)
try:
    block = bytearray(1024 * MIB)
    print("unexpected-allocation", len(block), flush=True)
except MemoryError:
    print("blocked", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "limit True" in proc.stdout
    assert "blocked" in proc.stdout
    assert "unexpected-allocation" not in proc.stdout


def test_wait_for_safe_memory_returns_without_failure_when_safe(monkeypatch):
    policy = MemoryPolicy.auto()

    class VM:
        available = policy.resume_system_bytes + GIB

    monkeypatch.setattr("oracle_lite.memory.psutil.virtual_memory", lambda: VM())
    events = []
    wait_for_safe_memory(
        policy,
        on_wait=lambda state: events.append(state),
        poll_seconds=0.0,
        stable_samples=1,
    )
    assert events
    assert events[-1]["safe"] is True

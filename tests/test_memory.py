import subprocess
import sys

import pytest

from oracle_lite.memory import (
    GIB,
    MIB,
    MEMORY_SCOPE_ENV,
    MemoryPolicy,
    build_systemd_memory_scope_command,
    ensure_linux_memory_scope,
)


def test_auto_memory_policy_keeps_conservative_system_reserve_and_worker_cap():
    policy = MemoryPolicy.auto()
    assert policy.total_bytes > 0
    assert policy.reserve_system_bytes >= 12 * GIB
    assert policy.reserve_system_bytes < policy.total_bytes
    assert 4 * GIB <= policy.max_worker_rss_bytes <= 8 * GIB
    assert 5 * GIB <= policy.max_worker_address_space_bytes <= 10 * GIB
    assert policy.max_worker_rss_bytes < policy.total_bytes
    assert policy.max_worker_address_space_bytes < policy.total_bytes
    assert policy.max_swap_growth_bytes == 512 * MIB
    assert policy.cgroup_memory_max_bytes == policy.total_bytes - policy.reserve_system_bytes
    assert policy.cgroup_swap_max_bytes == 512 * MIB


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


def test_systemd_scope_command_contains_hard_memory_limits(monkeypatch):
    import oracle_lite.memory as memory

    monkeypatch.setattr(
        memory.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(memory.sys, "executable", "/tmp/venv/bin/python")

    policy = MemoryPolicy.auto()
    command = build_systemd_memory_scope_command(
        policy,
        argv=["run", "--max-steps", "10"],
    )

    assert command[0] == "/usr/bin/systemd-run"
    assert "--user" in command
    assert "--scope" in command
    assert f"MemoryMax={policy.cgroup_memory_max_bytes}" in command
    assert f"MemorySwapMax={policy.cgroup_swap_max_bytes}" in command
    assert f"{MEMORY_SCOPE_ENV}=1" in command
    assert command[-3:] == ["run", "--max-steps", "10"]


def test_existing_scope_marker_prevents_recursive_reexec(monkeypatch):
    monkeypatch.setenv(MEMORY_SCOPE_ENV, "1")
    assert ensure_linux_memory_scope(MemoryPolicy.auto(), argv=["run"]) is False

from oracle_lite.memory import GIB, MemoryPolicy


def test_auto_memory_policy_keeps_system_reserve_and_worker_cap():
    policy = MemoryPolicy.auto()
    assert policy.total_bytes > 0
    assert policy.reserve_system_bytes >= 6 * GIB
    assert policy.reserve_system_bytes < policy.total_bytes
    assert 4 * GIB <= policy.max_worker_rss_bytes <= 10 * GIB
    assert policy.max_worker_rss_bytes < policy.total_bytes

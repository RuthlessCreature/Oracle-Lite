from types import SimpleNamespace
from pathlib import Path

from oracle_lite.resources import (
    GIB,
    HostResourcePolicy,
    recommended_pdf_max_pixels,
    scale_for_pixel_budget,
    wait_for_disk,
    wait_for_ram,
)


def test_resource_policy_is_conservative_on_64gb_1tb_host(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "oracle_lite.resources.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * GIB, available=48 * GIB),
    )
    monkeypatch.setattr(
        "oracle_lite.resources.shutil.disk_usage",
        lambda _: SimpleNamespace(
            total=1024 * GIB,
            used=200 * GIB,
            free=824 * GIB,
        ),
    )

    policy = HostResourcePolicy.auto(tmp_path)
    assert policy.ram_reserve_bytes >= 19 * GIB
    assert policy.ram_resume_bytes >= policy.ram_reserve_bytes + policy.parser_address_space_bytes
    assert policy.parser_address_space_bytes <= 6 * GIB
    assert policy.disk_reserve_bytes >= 80 * GIB
    assert policy.gpu_step_reserve_bytes >= int(2.5 * GIB)


def test_wait_gates_return_immediately_when_safe(monkeypatch, tmp_path: Path):
    policy = HostResourcePolicy(
        ram_total_bytes=64 * GIB,
        ram_reserve_bytes=20 * GIB,
        ram_resume_bytes=26 * GIB,
        parser_address_space_bytes=6 * GIB,
        parser_rss_budget_bytes=5 * GIB,
        disk_total_bytes=1000 * GIB,
        disk_reserve_bytes=100 * GIB,
        disk_resume_bytes=120 * GIB,
        gpu_step_reserve_bytes=int(2.5 * GIB),
        model_download_budget_bytes=30 * GIB,
        training_write_budget_bytes=10 * GIB,
    )
    monkeypatch.setattr(
        "oracle_lite.resources.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * GIB, available=40 * GIB),
    )
    monkeypatch.setattr(
        "oracle_lite.resources.shutil.disk_usage",
        lambda _: SimpleNamespace(
            total=1000 * GIB,
            used=500 * GIB,
            free=500 * GIB,
        ),
    )

    wait_for_ram(policy, poll_seconds=0.0, stable_samples=1)
    wait_for_disk(
        tmp_path,
        policy,
        required_bytes=30 * GIB,
        poll_seconds=0.0,
        stable_samples=1,
    )


def test_pdf_pixel_budget_is_bounded():
    assert recommended_pdf_max_pixels(0) == 1_048_576
    assert recommended_pdf_max_pixels(4) == 262_144

    scale = scale_for_pixel_budget(4000, 3000, 1_048_576)
    assert scale < 1.0
    assert 4000 * scale * 3000 * scale <= 1_048_576 + 1

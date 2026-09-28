import json
from pathlib import Path

from oracle_lite.config import AppConfig
from oracle_lite.db import Registry, utcnow
from oracle_lite.pipeline import run_one_click


def make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "output",
    )
    cfg.ensure_dirs()
    return cfg


def fake_trainer_factory(calls: list[tuple[str, int | None]]):
    def fake_trainer(cfg: AppConfig, *, snapshot_id: str, max_steps: int | None = None) -> str:
        calls.append((snapshot_id, max_steps))
        run_id = f"fake-{len(calls)}"
        output_dir = cfg.training_dir / snapshot_id / "fake"
        adapter_dir = output_dir / "adapter-final"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "run.json").write_text(
            json.dumps({
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "preset_values": {
                    "max_steps": -1 if max_steps is None else max_steps,
                },
            }),
            encoding="utf-8",
        )

        registry = Registry(cfg.registry_path)
        now = utcnow()
        with registry.connect() as conn:
            conn.execute(
                """INSERT INTO training_runs(
                       run_id,snapshot_id,kind,status,config_path,output_dir,started_at,ended_at
                   ) VALUES(?,?,'multimodal-domain','completed',NULL,?,?,?)""",
                (run_id, snapshot_id, str(output_dir), now, now),
            )
        return run_id

    return fake_trainer


def test_one_click_is_idempotent_until_corpus_changes(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    source = cfg.corpus_dir / "manual.txt"
    source.write_text("alpha domain knowledge", encoding="utf-8")

    calls: list[tuple[str, int | None]] = []
    trainer = fake_trainer_factory(calls)

    first = run_one_click(cfg, trainer=trainer)
    assert first.status == "trained"
    assert first.snapshot_reused is False
    assert len(calls) == 1

    second = run_one_click(cfg, trainer=trainer)
    assert second.status == "up_to_date"
    assert second.snapshot_id == first.snapshot_id
    assert second.snapshot_reused is True
    assert len(calls) == 1

    source.write_text("beta domain knowledge", encoding="utf-8")
    third = run_one_click(cfg, trainer=trainer)
    assert third.status == "trained"
    assert third.snapshot_id != first.snapshot_id
    assert third.snapshot_reused is False
    assert len(calls) == 2


def test_smoke_run_never_marks_snapshot_fully_trained(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    (cfg.corpus_dir / "manual.txt").write_text(
        "domain knowledge for smoke test",
        encoding="utf-8",
    )

    calls: list[tuple[str, int | None]] = []
    trainer = fake_trainer_factory(calls)

    smoke = run_one_click(cfg, max_steps=2, trainer=trainer)
    assert smoke.status == "smoke_trained"
    assert calls == [(smoke.snapshot_id, 2)]

    full = run_one_click(cfg, trainer=trainer)
    assert full.status == "trained"
    assert full.snapshot_id == smoke.snapshot_id
    assert calls[-1] == (smoke.snapshot_id, None)

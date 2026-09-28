import json
from pathlib import Path

from oracle_lite.config import AppConfig
from oracle_lite.ingest import ingest_corpus
from oracle_lite.scanner import scan_corpus
from oracle_lite.snapshot import create_snapshot


def make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "output",
    )
    cfg.ensure_dirs()
    return cfg


def manifest_hashes(path: Path) -> set[str]:
    return {
        json.loads(line)["content_hash"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_snapshot_is_immutable_when_source_changes(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    source = cfg.corpus_dir / "knowledge.txt"
    source.write_text("version one", encoding="utf-8")

    scan_corpus(cfg, verify_all=True)
    ingest_corpus(cfg)
    first = create_snapshot(cfg, name="v1", mode="full")
    first_hashes = manifest_hashes(first.manifest_path)

    source.write_text("version two", encoding="utf-8")
    scan_corpus(cfg, verify_all=True)
    ingest_corpus(cfg)
    second = create_snapshot(
        cfg,
        name="v2",
        mode="incremental",
        base_snapshot_id=first.snapshot_id,
        history_replay_ratio=0.0,
    )
    second_hashes = manifest_hashes(second.manifest_path)

    assert len(first_hashes) == 1
    assert len(second_hashes) == 1
    assert first_hashes != second_hashes
    assert manifest_hashes(first.manifest_path) == first_hashes

import json
from pathlib import Path

from oracle_lite.config import AppConfig
from oracle_lite.db import Registry
from oracle_lite.ingest import ingest_corpus
from oracle_lite.scanner import scan_corpus


def test_json_ingestion_creates_traceable_canonical_artifact(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "alarm.json"
    source.write_text(
        json.dumps({"alarm_code": "E102", "action": "check blockage"}),
        encoding="utf-8",
    )

    cfg = AppConfig(corpus_roots=[corpus], state_dir=tmp_path / ".oracle")
    cfg.ensure_dirs()
    scan_corpus(cfg, verify_all=True)
    stats = ingest_corpus(cfg)
    assert stats.ready == 1

    registry = Registry(cfg.registry_path)
    row = registry.list_active_unique_content()[0]
    artifact = registry.get_artifact(row["content_hash"], cfg.parser_version)
    assert artifact["status"] == "ready"

    canonical = json.loads(Path(artifact["canonical_path"]).read_text(encoding="utf-8"))
    assert canonical["content_hash"] == row["content_hash"]
    assert "alarm_code: E102" in canonical["text"]

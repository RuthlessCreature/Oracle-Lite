import json
from pathlib import Path

from oracle_lite.canonical import CanonicalDocument
from oracle_lite.config import AppConfig
from oracle_lite.db import Registry
from oracle_lite.ingest import ingest_corpus
from oracle_lite.ingest_worker import _stream_low_memory_canonical
from oracle_lite.scanner import scan_corpus


def test_json_ingestion_creates_traceable_canonical_artifact(tmp_path: Path):
    corpus = tmp_path / "corpus"
    source = corpus / "alarm.json"

    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=corpus,
        output_dir=tmp_path / "output",
    )
    cfg.ensure_dirs()
    source.write_text(
        json.dumps({"alarm_code": "E102", "action": "check blockage"}),
        encoding="utf-8",
    )

    scan_corpus(cfg, verify_all=True)
    stats = ingest_corpus(cfg)
    assert stats.ready == 1

    registry = Registry(cfg.registry_path)
    row = registry.list_active_unique_content()[0]
    artifact = registry.get_artifact(row["content_hash"], cfg.parser_version)
    assert artifact["status"] == "ready"

    canonical_path = Path(artifact["canonical_path"])
    header = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert header["content_hash"] == row["content_hash"]
    assert header["segments_path"]
    assert Path(header["segments_path"]).exists()

    canonical = CanonicalDocument.read_json(canonical_path)
    assert any("alarm_code: E102" in segment.text for segment in canonical.segments)


def test_low_memory_text_fallback_streams_sidecar(tmp_path: Path):
    source = tmp_path / "huge.txt"
    source.write_text("abc\n" * 300000, encoding="utf-8")
    canonical = tmp_path / "canonical.json"
    assets = tmp_path / "assets"

    result = _stream_low_memory_canonical(
        source=source,
        asset_dir=assets,
        canonical_path=canonical,
        content_hash="a" * 64,
        parser_version="v3-memory-safe",
        memory_level=2,
    )

    assert result["low_memory_mode"] is True
    header = CanonicalDocument.read_header(canonical)
    assert Path(header["segments_path"]).exists()
    segments = list(CanonicalDocument.iter_segments(canonical))
    assert len(segments) > 1
    assert sum(len(s.text) for s in segments) > 100000

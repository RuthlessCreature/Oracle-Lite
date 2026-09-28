import json
from pathlib import Path

import fitz
from PIL import Image

from oracle_lite.canonical import CanonicalDocument
from oracle_lite.config import AppConfig
from oracle_lite.db import Registry
from oracle_lite.ingest import ingest_corpus
from oracle_lite.scanner import scan_corpus


def make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "output",
    )
    cfg.ensure_dirs()
    return cfg


def canonical_for_only_source(cfg: AppConfig) -> CanonicalDocument:
    registry = Registry(cfg.registry_path)
    row = registry.list_active_unique_content()[0]
    artifact = registry.get_artifact(row["content_hash"], cfg.parser_version)
    return CanonicalDocument.read_json(artifact["canonical_path"])


def test_native_image_is_preserved_as_visual_asset(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    source = cfg.corpus_dir / "bearing_damage.png"
    Image.new("RGB", (64, 48), "white").save(source)

    scan_corpus(cfg, verify_all=True)
    stats = ingest_corpus(cfg)
    assert stats.visual_documents == 1

    doc = canonical_for_only_source(cfg)
    assert doc.segments[0].images
    assert Path(doc.segments[0].images[0]).exists()


def test_pdf_becomes_page_image_plus_page_text(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    source = cfg.corpus_dir / "manual.pdf"

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Alarm E102: check mechanical blockage.")
    pdf.save(source)
    pdf.close()

    scan_corpus(cfg, verify_all=True)
    stats = ingest_corpus(cfg)
    assert stats.visual_segments == 1

    doc = canonical_for_only_source(cfg)
    assert len(doc.segments) == 1
    assert "E102" in doc.segments[0].text
    assert len(doc.segments[0].images) == 1
    assert Path(doc.segments[0].images[0]).exists()

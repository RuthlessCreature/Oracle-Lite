import json
from pathlib import Path

import pymupdf as fitz
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


def test_large_pdf_page_is_rasterized_with_pixel_ceiling(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    source = cfg.corpus_dir / "large-page.pdf"

    pdf = fitz.open()
    page = pdf.new_page(width=4000, height=3000)
    page.insert_text((100, 100), "Large drawing page")
    pdf.save(source)
    pdf.close()

    scan_corpus(cfg, verify_all=True)
    stats = ingest_corpus(cfg)
    assert stats.failed == 0

    doc = canonical_for_only_source(cfg)
    assert doc.segments
    image_path = Path(doc.segments[0].images[0])
    with Image.open(image_path) as image:
        width, height = image.size
    assert width * height <= 1_048_576 + 4096


def test_pdf_table_fields_keep_geometric_row_relationships(tmp_path):
    from oracle_lite.pdf_layout import extract_pdf_text

    pdf = fitz.open()
    page = pdf.new_page()
    labels = [('Pressure', '12 kPa'), ('Temperature', '90 C'), ('Revision', 'B')]
    # Labels and values are written in separate passes, as in many exported PDFs.
    for i, (label, value) in enumerate(labels):
        page.insert_text((40, 50 + i * 24), label)
    for i, (label, value) in enumerate(labels):
        page.insert_text((200, 50 + i * 24), value)
    text = extract_pdf_text(page)
    lines = text.splitlines()
    assert len(lines) == 3
    for line, (label, value) in zip(lines, labels):
        assert label in line and value in line
    pdf.close()


def test_pdf_positioned_blocks_keep_identifiers_separate_from_measurements():
    from oracle_lite.pdf_layout import extract_pdf_positioned_blocks

    pdf = fitz.open()
    page = pdf.new_page(width=400, height=300)
    page.insert_text((40, 60), "8 SPOTS PER NUT")
    page.insert_text((40, 74), "MAX WELD HEIGHT 0.05")
    page.insert_text((230, 60), "FAI")
    page.insert_text((231, 72), "54")
    blocks = extract_pdf_positioned_blocks(page)
    identifiers = [block for block in blocks if block["kind"] == "identifier"]
    measurements = [block for block in blocks if block["kind"] == "text"]

    assert any("FAI 54" in block["text"] for block in identifiers)
    assert any("8 SPOTS PER NUT" in block["text"] and "0.05" in block["text"]
               for block in measurements)
    assert not any("FAI 54" in block["text"] for block in measurements)
    assert all(0 <= block["x"] <= 100 and 0 <= block["y"] <= 100 for block in blocks)
    pdf.close()

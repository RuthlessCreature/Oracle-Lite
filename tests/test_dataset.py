import json
from pathlib import Path

from docx import Document
from PIL import Image

from oracle_lite.config import AppConfig
from oracle_lite.dataset import build_domain_dataset
from oracle_lite.ingest import ingest_corpus
from oracle_lite.scanner import scan_corpus
from oracle_lite.snapshot import create_snapshot


def test_multi_image_segment_is_split_to_one_image_per_training_record(tmp_path: Path):
    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "output",
    )
    cfg.ensure_dirs()

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (32, 32), "white").save(image_a)
    Image.new("RGB", (32, 32), "black").save(image_b)

    source = cfg.corpus_dir / "manual.docx"
    doc = Document()
    doc.add_paragraph("Bearing inspection reference")
    doc.add_picture(str(image_a))
    doc.add_picture(str(image_b))
    doc.save(source)

    scan_corpus(cfg, verify_all=True)
    stats = ingest_corpus(cfg)
    assert stats.failed == 0

    snap = create_snapshot(cfg, name="dataset-memory-test", mode="full")
    dataset = build_domain_dataset(cfg, snapshot_id=snap.snapshot_id)

    records = []
    for shard in dataset.shard_paths:
        with shard.open("r", encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())

    visual = [
        row for row in records
        if row["mode"] == "multimodal" and row["source_path"].endswith("manual.docx")
    ]
    assert len(visual) == 2
    assert all(len(row["images"]) == 1 for row in visual)

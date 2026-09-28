from pathlib import Path

from oracle_lite.config import AppConfig
from oracle_lite.db import Registry
from oracle_lite.scanner import scan_corpus


def make_cfg(tmp_path: Path) -> AppConfig:
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=corpus,
        output_dir=output,
    )
    cfg.ensure_dirs()
    return cfg


def test_scan_tracks_change_and_tombstone(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    corpus = cfg.corpus_dir
    p = corpus / "a.txt"
    p.write_text("alpha", encoding="utf-8")

    first = scan_corpus(cfg, verify_all=True)
    assert first.new == 1
    assert first.tombstoned == 0

    second = scan_corpus(cfg, verify_all=True)
    assert second.unchanged == 1

    p.write_text("beta", encoding="utf-8")
    third = scan_corpus(cfg, verify_all=True)
    assert third.changed == 1

    p.unlink()
    fourth = scan_corpus(cfg, verify_all=True)
    assert fourth.tombstoned == 1


def test_same_content_is_deduplicated_by_hash(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    corpus = cfg.corpus_dir
    (corpus / "a.txt").write_text("same", encoding="utf-8")
    (corpus / "b.txt").write_text("same", encoding="utf-8")

    stats = scan_corpus(cfg, verify_all=True)
    assert stats.files_seen == 2
    assert stats.duplicates == 1

    registry = Registry(cfg.registry_path)
    assert len(registry.list_active_unique_content()) == 1


def test_rename_keeps_one_content_object(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    corpus = cfg.corpus_dir
    old = corpus / "old.txt"
    old.write_text("payload", encoding="utf-8")
    scan_corpus(cfg, verify_all=True)

    new = corpus / "new.txt"
    old.rename(new)
    stats = scan_corpus(cfg, verify_all=True)

    assert stats.tombstoned == 1
    registry = Registry(cfg.registry_path)
    assert len(registry.list_active_unique_content()) == 1

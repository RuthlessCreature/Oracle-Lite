from pathlib import Path

from oracle_lite.config import AppConfig
from oracle_lite.ingest import ingest_corpus
from oracle_lite.scanner import scan_corpus


def make_cfg(tmp_path: Path, corpus_name: str) -> AppConfig:
    cfg = AppConfig(
        minimax_api_key="sk-cp-test",
        corpus_dir=tmp_path / corpus_name,
        output_dir=tmp_path / "shared-output",
    )
    cfg.ensure_dirs()
    return cfg


def test_corpus_directory_has_strict_namespace_isolation(tmp_path: Path):
    a = make_cfg(tmp_path, "corpus-a")
    b = make_cfg(tmp_path, "corpus-b")
    assert a.corpus_id != b.corpus_id
    assert a.registry_path != b.registry_path
    assert a.snapshots_dir != b.snapshots_dir
    assert a.datasets_dir != b.datasets_dir
    assert a.training_dir != b.training_dir
    assert a.models_dir == b.models_dir

    (a.corpus_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (b.corpus_dir / "b.txt").write_text("beta", encoding="utf-8")
    assert scan_corpus(a).new == 1
    assert scan_corpus(b).new == 1
    assert scan_corpus(a).hashed == 0
    assert scan_corpus(b).hashed == 0


def test_step_is_first_class_corpus_source(tmp_path: Path):
    cfg = make_cfg(tmp_path, "cad")
    step = cfg.corpus_dir / "bracket.step"
    step.write_text(
        """ISO-10303-21;
HEADER;
FILE_SCHEMA (('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=ADVANCED_FACE('',(),$,.T.);
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )
    stats = scan_corpus(cfg, verify_all=True)
    assert stats.files_seen == 1
    ingested = ingest_corpus(cfg)
    assert ingested.failed == 0
    assert ingested.ready == 1

    originals = list(cfg.assets_dir.rglob("source.step"))
    assert len(originals) == 1
    canonical = list(cfg.canonical_dir.rglob("*.json"))
    assert len(canonical) == 1
    text = canonical[0].read_text(encoding="utf-8")
    assert "CONFIG_CONTROL_DESIGN" in text
    assert "CARTESIAN_POINT" in text
    assert "geometry_preserved" in text


def test_parasolid_extensions_are_scanned(tmp_path: Path):
    cfg = make_cfg(tmp_path, "parasolid")
    (cfg.corpus_dir / "part.x_t").write_text(
        ": TRANSMIT FILE created by modeller version 3800235\nSCH_3800235_22000\nBODY FACE EDGE\n",
        encoding="latin-1",
    )
    (cfg.corpus_dir / "part.x_b").write_bytes(b"\x00\x01opaque parasolid binary")
    stats = scan_corpus(cfg, verify_all=True)
    assert stats.files_seen == 2
    ingested = ingest_corpus(cfg)
    assert ingested.failed == 0
    assert ingested.ready == 2

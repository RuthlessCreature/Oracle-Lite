from pathlib import Path

import pytest

from oracle_lite.config import load_config


def test_three_field_config_derives_internal_paths(tmp_path: Path):
    config = tmp_path / "oracle.yaml"
    config.write_text(
        'minimax_api_key: "sk-cp-test"\n'
        'corpus_dir: "./corpus"\n'
        'output_dir: "./output"\n',
        encoding="utf-8",
    )

    cfg = load_config(config)
    assert cfg.minimax_api_key == "sk-cp-test"
    assert cfg.corpus_dir == (tmp_path / "corpus").resolve()
    assert cfg.registry_path == (
        tmp_path / "output" / "_corpora" / cfg.corpus_id / "_state" / "registry.sqlite3"
    ).resolve()
    assert cfg.datasets_dir == (tmp_path / "output" / "datasets").resolve()


def test_unknown_config_key_is_rejected(tmp_path: Path):
    config = tmp_path / "oracle.yaml"
    config.write_text(
        'minimax_api_key: "x"\n'
        'corpus_dir: "./corpus"\n'
        'output_dir: "./output"\n'
        'learning_rate: 0.01\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only"):
        load_config(config)


def test_output_cannot_live_inside_corpus(tmp_path: Path):
    config = tmp_path / "oracle.yaml"
    config.write_text(
        'minimax_api_key: "x"\n'
        'corpus_dir: "./corpus"\n'
        'output_dir: "./corpus/generated"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not"):
        load_config(config)

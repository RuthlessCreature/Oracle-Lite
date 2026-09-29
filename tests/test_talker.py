from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from talker.attachments import parse_attachment
from talker.config import load_config
from talker.history import ChatHistory
from talker.resolver import discover_latest_adapter


def test_talker_config_has_single_training_output_field(tmp_path: Path):
    cfg_file = tmp_path / "talker.yaml"
    cfg_file.write_text('training_output_dir: "./training-root"\n', encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.training_output_dir == (tmp_path / "training-root").resolve()
    assert cfg.history_db.name == "history.sqlite3"
    assert cfg.uploads_dir.exists()
    assert cfg.parsed_dir.exists()


def test_talker_config_rejects_unknown_keys(tmp_path: Path):
    cfg_file = tmp_path / "talker.yaml"
    cfg_file.write_text(
        'training_output_dir: "./x"\nport: 9999\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only training_output_dir"):
        load_config(cfg_file)


def test_chat_history_persists_messages_and_attachments(tmp_path: Path):
    db = tmp_path / "history.sqlite3"
    history = ChatHistory(db)
    conversation = history.new_conversation()
    user_id = history.add_message(conversation["id"], "user", "check drawing")
    history.add_attachment(
        user_id,
        filename="part.step",
        stored_path="/tmp/part.step",
        mime_type="application/octet-stream",
        parsed_text="Entity count: 10",
        images=[],
    )
    history.add_message(conversation["id"], "assistant", "ok")

    reopened = ChatHistory(db)
    messages = reopened.get_messages(conversation["id"])
    assert [row["role"] for row in messages] == ["user", "assistant"]
    assert messages[0]["attachments"][0]["filename"] == "part.step"
    assert messages[0]["attachments"][0]["parsed_text"] == "Entity count: 10"


def test_adapter_resolver_finds_local_base_and_latest_adapter(tmp_path: Path):
    root = tmp_path / "output"
    base = root / "models" / "Qwen3.5-9B-Base"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}", encoding="utf-8")

    run1 = root / "_corpora" / "a" / "training" / "snap1" / "run"
    final = run1 / "adapter-final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run1 / "run.json").write_text(
        json.dumps({"base_model_path": str(base)}),
        encoding="utf-8",
    )

    run2 = root / "_corpora" / "a" / "training" / "snap2" / "run"
    checkpoint = run2 / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run2 / "run.json").write_text(
        json.dumps({"base_model_path": str(base)}),
        encoding="utf-8",
    )

    os.utime(final / "adapter_config.json", (1000, 1000))
    os.utime(checkpoint / "adapter_config.json", (2000, 2000))

    bundle = discover_latest_adapter(root)
    assert bundle.adapter_dir == checkpoint.resolve()
    assert bundle.base_model_path == base.resolve()
    assert bundle.source_kind == "checkpoint"


def test_attachment_parser_handles_text_and_step(tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("bearing preload 12 N", encoding="utf-8")
    text, images = parse_attachment(text_file, tmp_path / "assets-text")
    assert "bearing preload 12 N" in text
    assert images == []

    step = tmp_path / "part.step"
    step.write_text(
        """ISO-10303-21;
HEADER;
FILE_SCHEMA (('AP214'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=ADVANCED_FACE('',(),$,.T.);
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )
    text, images = parse_attachment(step, tmp_path / "assets-step")
    assert "AP214" in text
    assert "CARTESIAN_POINT" in text
    assert images == []


def test_talker_web_asset_contains_chat_history_and_upload_controls():
    html = Path("talker/static/index.html").read_text(encoding="utf-8")
    assert 'id="conversations"' in html
    assert 'id="files"' in html
    assert 'id="sendBtn"' in html
    assert "/api/chat" in html
    assert "/api/reload-model" in html


def test_talker_entrypoint_and_versions():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    oracle_version = Path("oracle_lite/__init__.py").read_text(encoding="utf-8")
    talker_version = Path("talker/__init__.py").read_text(encoding="utf-8")
    assert 'version = "0.6.2"' in pyproject
    assert 'oracle-talker = "talker.__main__:main"' in pyproject
    assert 'include = ["oracle_lite*", "talker*"]' in pyproject
    assert '__version__ = "0.6.2"' in oracle_version
    assert '__version__ = "0.6.2"' in talker_version


def test_long_answers_continue_without_total_token_cap(monkeypatch, tmp_path: Path):
    from talker.model import TalkerModel

    talker = TalkerModel(tmp_path)
    calls = []

    def fake_generate(messages, **kwargs):
        calls.append(kwargs)
        index = len(calls)
        if index < 5:
            return f"FAI{index:02d}\n", True, 768
        return "FAI05 complete", False, 7

    monkeypatch.setattr(talker, "_generate_once", fake_generate)
    answer = talker._answer_with_auto_continue(
        [{"role": "user", "content": "列出所有FAI", "attachments": []}],
        max_context_chars=1000,
        chunk_tokens=768,
        max_images=0,
    )
    assert "FAI01" in answer
    assert "FAI05 complete" in answer
    assert len(calls) == 5
    assert "total_tokens" not in calls[0]


def test_unbounded_continuation_stops_on_repeated_chunk(monkeypatch, tmp_path: Path):
    from talker.model import TalkerModel

    talker = TalkerModel(tmp_path)
    calls = []

    def fake_generate(messages, **kwargs):
        calls.append(kwargs)
        return "same repeated chunk", True, 768

    monkeypatch.setattr(talker, "_generate_once", fake_generate)
    answer = talker._answer_with_auto_continue(
        [{"role": "user", "content": "继续", "attachments": []}],
        max_context_chars=1000,
        chunk_tokens=768,
        max_images=0,
    )
    assert answer == "same repeated chunk"
    assert len(calls) == 2


def test_talker_no_longer_has_384_token_hard_cut_and_supports_lan_access():
    model = Path("talker/model.py").read_text(encoding="utf-8")
    main = Path("talker/__main__.py").read_text(encoding="utf-8")
    assert "GENERATION_TOTAL_TOKENS" not in model
    assert "GENERATION_FALLBACK_TOTAL_TOKENS" not in model
    assert "GENERATION_CHUNK_TOKENS = 768" in model
    assert "CONTINUATION_TAIL_CHARS = 8_000" in model
    assert 'max_new_tokens=384' not in model
    assert 'host="0.0.0.0"' in main
    assert 'sock.bind(("0.0.0.0", port))' in main
    assert "_lan_ipv4_addresses" in main

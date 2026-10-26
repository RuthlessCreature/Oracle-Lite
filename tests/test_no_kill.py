from pathlib import Path


def test_core_has_no_active_process_kill_calls():
    root = Path("oracle_lite")
    forbidden = (
        ".terminate(",
        ".kill(",
        "signal.SIGKILL",
        "os.kill(",
    )
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append((str(path), token))
    assert offenders == []


def test_training_uses_streaming_dataset(tmp_path):
    import json
    import pytest
    from torch.utils.data import IterableDataset
    from oracle_lite.training import load_training_dataset

    path = tmp_path / "rows.jsonl"
    dataset = load_training_dataset([path])
    assert isinstance(dataset, IterableDataset)
    # Dataset creation does not read ahead or materialize the input.
    path.write_text('{"text":"first record","images":[]}\nnot valid JSON\n')
    iterator = iter(dataset)
    assert next(iterator)["text"] == "first record"
    with pytest.raises(json.JSONDecodeError):
        next(iterator)

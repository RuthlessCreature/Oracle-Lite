from pathlib import Path


def test_core_has_no_active_process_kill_calls():
    root = Path("oracle_lite")
    forbidden = (
        ".terminate(",
        ".kill(",
        "SIGKILL",
        "os.kill(",
    )
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append((str(path), token))
    assert offenders == []


def test_training_uses_streaming_dataset():
    text = Path("oracle_lite/training.py").read_text(encoding="utf-8")
    assert "streaming=True" in text
    assert "load_dataset(" in text
    assert "CUDA OOM caught; no process exit" in text

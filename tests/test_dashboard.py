import json
import urllib.request
from pathlib import Path

from oracle_lite.dashboard import TrainingDashboard, TrainingMonitor


def test_monitor_state_and_final_report(tmp_path: Path):
    monitor = TrainingMonitor(tmp_path / "output")
    monitor.update_phase("scan", "Scanning corpus")
    monitor.update("corpus", {"files_seen": 12, "new": 3})
    monitor.update_training(step=4, total_steps=10, progress_percent=40.0, loss=1.23)
    monitor.log("INFO", "hello")

    state = monitor.snapshot()
    assert state["phase"] == "scan"
    assert state["corpus"]["files_seen"] == 12
    assert state["training"]["step"] == 4
    assert state["logs"][-1]["message"] == "hello"

    final_path = monitor.save_final_state()
    saved = json.loads(final_path.read_text(encoding="utf-8"))
    assert saved["training"]["progress_percent"] == 40.0
    assert Path(saved["log_file"]).exists()


def test_dashboard_serves_html_and_state_api(tmp_path: Path):
    monitor = TrainingMonitor(tmp_path / "output")
    dashboard = TrainingDashboard(
        monitor,
        preferred_port=0,
        open_browser=False,
    )
    url = dashboard.start()
    try:
        with urllib.request.urlopen(url + "health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload == {"ok": True}

        monitor.update_phase("training", "Training model")
        monitor.update_training(step=2, total_steps=8, progress_percent=25.0)

        with urllib.request.urlopen(url + "api/state", timeout=3) as response:
            state = json.loads(response.read().decode("utf-8"))
        assert state["phase"] == "training"
        assert state["training"]["step"] == 2

        with urllib.request.urlopen(url, timeout=3) as response:
            html = response.read().decode("utf-8")
        assert "ORACLE-LITE / TRAINING CONSOLE" in html
        assert "GPU / VRAM" in html
        assert "Live Logs" in html
    finally:
        dashboard.stop()

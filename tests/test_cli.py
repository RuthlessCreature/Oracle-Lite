from pathlib import Path


def test_one_click_cli_has_no_removed_watchdog_reference():
    text = Path("oracle_lite/cli.py").read_text(encoding="utf-8")
    assert "watchdog.stop()" not in text
    assert "HostMemoryWatchdog" not in text

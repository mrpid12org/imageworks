import logging

from imageworks.logging_utils import configure_logging


def test_configure_logging_honours_env_override(monkeypatch, tmp_path):
    target_dir = tmp_path / "logs"
    monkeypatch.setenv("IMAGEWORKS_LOG_DIR", str(target_dir))

    log_path = configure_logging("unit_test", include_console=False)
    logging.getLogger(__name__).info("env override works")

    assert log_path == target_dir / "unit_test.log"
    assert log_path.exists()
    assert "env override works" in log_path.read_text()

    monkeypatch.delenv("IMAGEWORKS_LOG_DIR")


def test_configure_logging_replaces_previous_handlers(tmp_path):
    first_dir = tmp_path / "logs"
    second_dir = tmp_path / "alt_logs"

    first_path = configure_logging("first_run", log_dir=first_dir, include_console=False)
    logging.getLogger(__name__).info("first run entry")
    assert first_path == first_dir / "first_run.log"
    assert "first run entry" in first_path.read_text()

    second_path = configure_logging("second_run", log_dir=second_dir, include_console=False)
    logging.getLogger(__name__).info("second run entry")
    assert second_path == second_dir / "second_run.log"
    assert "second run entry" in second_path.read_text()

    # Ensure the first log is not appended to after reconfiguration
    assert "second run entry" not in first_path.read_text()

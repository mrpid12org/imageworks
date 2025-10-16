import json
from imageworks.chat_proxy.logging_utils import JsonlLogger


def test_jsonl_logger_rotates(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "proxy.jsonl"
    logger = JsonlLogger(str(log_file), max_bytes=5, retention_days=1)

    # Deterministic timestamp for rotation target
    monkeypatch.setattr(
        "imageworks.chat_proxy.logging_utils.time.strftime",
        lambda *_: "19700101-000000",
    )

    logger.log({"a": 1})
    assert log_file.exists()

    # Force size over threshold to trigger rotation
    logger.log({"payload": "123456"})

    rotated = log_file.with_name(log_file.name + ".19700101-000000")
    assert rotated.exists(), "Rotated file missing"
    assert log_file.exists(), "Logger must create a new active file"
    with open(log_file, encoding="utf-8") as fh:
        content = fh.read().strip()
        assert content, "New log file should contain latest entry"
        json.loads(content)


def test_jsonl_logger_handles_missing_directory(tmp_path):
    log_file = tmp_path / "missing" / "proxy.jsonl"
    logger = JsonlLogger(str(log_file), max_bytes=100)
    logger.log({"event": "ok"})
    assert log_file.exists()

"""Stage 1 (IQA) entry point for containerised execution."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from imageworks.apps.judge_vision.config import JudgeVisionConfig
from imageworks.apps.judge_vision.runner import JudgeVisionRunner
from imageworks.logging_utils import configure_logging
from imageworks.tools.model_downloader.config import get_config


def _load_config(payload_path: Path) -> JudgeVisionConfig:
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    config = JudgeVisionConfig.from_dict(data)
    if (config.stage or "").lower() != "iqa":
        config.stage = "iqa"
    return config


def main(config_path: Optional[str] = None) -> None:
    """Entry point executed inside the TensorFlow container."""

    log_path = configure_logging("judge_vision")
    logger = logging.getLogger(__name__)
    logger.info("Judge Vision Stage 1 entry (log: %s)", log_path)
    logger.info(
        "Stage 1 model root: env=%s config=%s",
        os.environ.get("IMAGEWORKS_MODEL_ROOT"),
        get_config().linux_wsl.root,
    )

    if not config_path:
        raise SystemExit("Configuration path argument is required.")

    payload = Path(config_path)
    if not payload.exists():
        raise SystemExit(f"Configuration file not found: {payload}")

    config = _load_config(payload)
    runner = JudgeVisionRunner(config)
    runner.run()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_path)

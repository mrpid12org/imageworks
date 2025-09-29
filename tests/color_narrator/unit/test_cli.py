"""Tests for color-narrator CLI functionality.

Tests CLI command parsing, parameter validation, dry-run mode,
and integration with core processing modules.
"""

import json
import pytest
import typer
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from imageworks.apps.color_narrator.core.data_loader import ColorNarratorItem
from imageworks.apps.color_narrator.core.narrator import ProcessingResult
from imageworks.apps.color_narrator.core.vlm import VLMBackend, VLMResponse

from imageworks.apps.color_narrator.cli import main as cli_main
from imageworks.apps.color_narrator.cli.main import app


class TestColorNarratorCLI:
    """Test cases for color-narrator CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories for CLI testing."""
        images_dir = tmp_path / "images"
        overlays_dir = tmp_path / "overlays"
        images_dir.mkdir()
        overlays_dir.mkdir()

        mono_jsonl = tmp_path / "mono.jsonl"
        mono_jsonl.write_text('{"filename": "test.jpg", "contamination_level": 0.5}\n')

        return {
            "images_dir": images_dir,
            "overlays_dir": overlays_dir,
            "mono_jsonl": mono_jsonl,
        }

    @pytest.fixture
    def invoke_narrate(self, cli_runner):
        """Helper to invoke the narrate command with patched dependencies."""

        def _invoke(args, config=None, process_results=None):
            config = config or {}
            default_results = Path("tests/test_output/test_cli_results.jsonl")
            config.setdefault("narrate_results_path", str(default_results))
            default_results.parent.mkdir(parents=True, exist_ok=True)
            if process_results is None:
                process_results = []

            summary_path = Path("tests/test_output/test_cli_summary.md")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            args_with_summary = list(args)
            if "--summary" not in args_with_summary and "-s" not in args_with_summary:
                args_with_summary.extend(["--summary", str(summary_path)])

            # Ensure tests use deterministic sample assets
            if not any(arg in args_with_summary for arg in {"--debug", "--no-debug"}):
                args_with_summary.append("--debug")

            with (
                patch(
                    "imageworks.apps.color_narrator.cli.main.load_config",
                    return_value=config,
                ),
                patch(
                    "imageworks.apps.color_narrator.cli.main.ColorNarrator"
                ) as mock_narrator,
            ):
                instance = mock_narrator.return_value
                instance.metadata_writer = MagicMock()
                instance.metadata_writer.backup_original = True
                instance.process_all.return_value = process_results

                return cli_runner.invoke(app, ["narrate", *args_with_summary])

        return _invoke

    def test_cli_app_help(self, cli_runner):
        """Test CLI app help message."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Color-Narrator" in result.output
        assert "VLM-guided color localization" in result.output
        assert "compare-prompts" in result.output

    def test_narrate_command_help(self, cli_runner):
        """Test narrate command help message."""
        result = cli_runner.invoke(app, ["narrate", "--help"])

        assert result.exit_code == 0
        assert "Generate colour narration metadata" in result.output
        assert "--images" in result.output
        assert "--overlays" in result.output
        assert "--mono-jsonl" in result.output
        assert "--batch-size" in result.output
        assert "--no-meta" in result.output
        assert "--results-json" in result.output
        assert "--debug" in result.output
        assert "--vlm-backend" in result.output

    def test_validate_command_help(self, cli_runner):
        """Test validate command help message."""
        result = cli_runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate existing color narrations" in result.output
        assert "--images" in result.output
        assert "--mono-jsonl" in result.output

    def test_narrate_command_basic_execution(self, temp_dirs, invoke_narrate):
        """Test basic narrate command execution (skeleton implementation)."""
        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ]
        )

        assert result.exit_code == 0
        assert "Color Narrator — generating competition metadata" in result.output
        assert "✅ Color narration complete" in result.output

    def test_narrate_command_no_meta(self, temp_dirs, invoke_narrate):
        """Test narrate command with no-meta flag."""
        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--no-meta",
            ]
        )

        assert result.exit_code == 0
        assert "🔍 No-meta mode: files will not be modified" in result.output

    def test_narrate_command_with_batch_size(self, temp_dirs, invoke_narrate):
        """Test narrate command with custom batch size."""
        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--batch-size",
                "8",
            ]
        )

        assert result.exit_code == 0
        assert "🧮 Batch size: 8" in result.output

    def test_narrate_command_with_debug(self, temp_dirs, invoke_narrate):
        """Test narrate command with debug flag."""
        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--debug",
            ]
        )

        assert result.exit_code == 0
        assert "Color Narrator — generating competition metadata" in result.output

    def test_narrate_writes_results_json(self, temp_dirs, invoke_narrate, tmp_path):
        """Test that narrate writes JSONL results with prompt information."""

        results_path = tmp_path / "narrate_results.jsonl"
        summary_path = tmp_path / "narrate_summary.md"

        process_results = [
            ProcessingResult(
                item=ColorNarratorItem(
                    image_path=temp_dirs["images_dir"] / "test1.jpg",
                    overlay_path=temp_dirs["overlays_dir"] / "test1.png",
                    mono_data={"verdict": "fail", "mode": "toned"},
                ),
                vlm_response=VLMResponse(
                    description="Sample narration",
                    confidence=0.9,
                    color_regions=["test"],
                    processing_time=1.2,
                ),
                metadata_written=False,
                error=None,
                processing_time=1.2,
            )
        ]

        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--summary",
                str(summary_path),
                "--results-json",
                str(results_path),
            ],
            process_results=process_results,
        )

        assert result.exit_code == 0
        assert results_path.exists()
        content = results_path.read_text(encoding="utf-8").strip()
        assert content, "Results JSONL should not be empty"
        record = json.loads(content.splitlines()[0])
        assert record["prompt_id"] is not None
        assert record["vlm_model"]

    def test_validate_command_basic_execution(self, cli_runner, temp_dirs):
        """Test basic validate command execution (skeleton implementation)."""
        result = cli_runner.invoke(
            app,
            [
                "validate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ],
        )

        assert result.exit_code == 0
        assert "🔍 Color-Narrator - Validate command" in result.output
        assert "✅ Validation complete" in result.output

    def test_narrate_command_no_arguments(self, invoke_narrate):
        """Test narrate command with no arguments (uses debug fallback data)."""
        result = invoke_narrate([])

        assert result.exit_code == 0
        assert "Color Narrator — generating competition metadata" in result.output

    def test_validate_command_no_arguments(self, cli_runner):
        """Validate should succeed using default configuration when no paths provided."""
        result = cli_runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "🔍 Color-Narrator - Validate command" in result.output
        assert "✅ Validation complete" in result.output

    def test_narrate_command_short_flags(self, temp_dirs, invoke_narrate):
        """Test narrate command with short flag variants."""
        result = invoke_narrate(
            [
                "-i",
                str(temp_dirs["images_dir"]),
                "-o",
                str(temp_dirs["overlays_dir"]),
                "-j",
                str(temp_dirs["mono_jsonl"]),
                "-b",
                "2",
            ]
        )

        assert result.exit_code == 0
        assert "🧮 Batch size: 2" in result.output
        assert "Summary: FAIL=0  QUERY=0  TOTAL=0" in result.output

    def test_validate_command_short_flags(self, cli_runner, temp_dirs):
        """Test validate command with short flag variants."""
        result = cli_runner.invoke(
            app,
            [
                "validate",
                "-i",
                str(temp_dirs["images_dir"]),
                "-j",
                str(temp_dirs["mono_jsonl"]),
            ],
        )

        assert result.exit_code == 0
        assert "🔍 Color-Narrator - Validate command" in result.output

    def test_invalid_command(self, cli_runner):
        """Test invalid command handling."""
        result = cli_runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0
        # Typer will show usage/error message for invalid commands

    def test_narrate_command_pathlib_conversion(self, temp_dirs, invoke_narrate):
        """Test that string paths are properly converted to Path objects."""
        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ]
        )

        assert result.exit_code == 0
        assert "Summary: FAIL=0  QUERY=0  TOTAL=0" in result.output

    def test_app_direct_execution(self):
        """Test app can be executed directly."""
        # Test that the app can be called (this simulates python -m ... execution)
        assert app is not None
        assert isinstance(app, typer.Typer)

    def test_command_descriptions_present(self, cli_runner):
        """Test that commands have proper descriptions."""
        # Test narrate command has description
        result = cli_runner.invoke(app, ["narrate", "--help"])
        assert "Generate colour narration metadata" in result.output

        # Test validate command has description
        result = cli_runner.invoke(app, ["validate", "--help"])
        assert "Validate existing color narrations" in result.output

    def test_command_examples_present(self, cli_runner):
        """Test that commands include usage examples."""
        result = cli_runner.invoke(app, ["narrate", "--help"])

        # Check for example usage
        assert "Examples:" in result.output
        assert "imageworks-color-narrator narrate" in result.output

    @pytest.mark.parametrize("flag", ["--no-meta", "--debug"])
    def test_boolean_flags(self, temp_dirs, flag, invoke_narrate):
        """Test boolean flag handling."""
        result = invoke_narrate(
            [
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                flag,
            ]
        )

        assert result.exit_code == 0
        if flag == "--dry-run":
            assert "🔍 Dry run mode: no files will be modified" in result.output
        assert "Summary: FAIL=0  QUERY=0  TOTAL=0" in result.output

    def test_batch_size_validation(self, temp_dirs, invoke_narrate, cli_runner):
        """Test batch size parameter validation."""
        # Valid batch size
        result = invoke_narrate(
            [
                "--batch-size",
                "4",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ]
        )
        assert result.exit_code == 0
        assert "Summary: FAIL=0  QUERY=0  TOTAL=0" in result.output

        # Invalid batch size (should be handled by typer)
        result = cli_runner.invoke(app, ["narrate", "--batch-size", "invalid"])
        assert result.exit_code != 0


class TestVLMConfigResolution:
    """Tests for resolving runtime VLM settings."""

    def test_resolve_lmdeploy_overrides(self):
        cfg = {
            "vlm_backend": "vllm",
            "vlm_lmdeploy_base_url": "http://custom:23333/v1",
            "vlm_lmdeploy_model": "custom/lmdeploy",
            "vlm_lmdeploy_api_key": "token",
        }

        (
            backend,
            base_url,
            model,
            timeout,
            api_key,
            options,
        ) = cli_main._resolve_vlm_runtime_settings(
            cfg,
            backend="lmdeploy",
            timeout=300,
        )

        assert backend is VLMBackend.LMDEPLOY
        assert base_url == "http://custom:23333/v1"
        assert model == "custom/lmdeploy"
        assert api_key == "token"
        assert timeout == 300
        assert options is None

    def test_resolve_defaults(self):
        cfg = {}

        (
            backend,
            base_url,
            model,
            timeout,
            api_key,
            options,
        ) = cli_main._resolve_vlm_runtime_settings(cfg)

        assert backend is VLMBackend.LMDEPLOY
        assert base_url == "http://localhost:24001/v1"
        assert model == "Qwen2.5-VL-7B-AWQ"
        assert timeout == cli_main.DEFAULT_VLM_SETTINGS[VLMBackend.LMDEPLOY]["timeout"]
        assert api_key == "EMPTY"
        assert options is None

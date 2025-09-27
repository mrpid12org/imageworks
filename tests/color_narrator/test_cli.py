"""Tests for color-narrator CLI functionality.

Tests CLI command parsing, parameter validation, dry-run mode,
and integration with core processing modules.
"""

import pytest
import typer
from typer.testing import CliRunner

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

    def test_cli_app_help(self, cli_runner):
        """Test CLI app help message."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Color-Narrator" in result.output
        assert "VLM-guided color localization" in result.output

    def test_narrate_command_help(self, cli_runner):
        """Test narrate command help message."""
        result = cli_runner.invoke(app, ["narrate", "--help"])

        assert result.exit_code == 0
        assert "Generate natural language color descriptions" in result.output
        assert "--images" in result.output
        assert "--overlays" in result.output
        assert "--mono-jsonl" in result.output
        assert "--batch-size" in result.output
        assert "--dry-run" in result.output
        assert "--debug" in result.output

    def test_validate_command_help(self, cli_runner):
        """Test validate command help message."""
        result = cli_runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate existing color narrations" in result.output
        assert "--images" in result.output
        assert "--mono-jsonl" in result.output

    def test_narrate_command_basic_execution(self, cli_runner, temp_dirs):
        """Test basic narrate command execution (skeleton implementation)."""
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ],
        )

        assert result.exit_code == 0
        assert "🎨 Color-Narrator - Narrate command" in result.output
        assert (
            "⚠️  Full implementation coming soon - basic validation complete"
            in result.output
        )

    def test_narrate_command_dry_run(self, cli_runner, temp_dirs):
        """Test narrate command with dry-run flag."""
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "🔍 DRY RUN MODE - No files will be modified" in result.output

    def test_narrate_command_with_batch_size(self, cli_runner, temp_dirs):
        """Test narrate command with custom batch size."""
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--batch-size",
                "8",
            ],
        )

        assert result.exit_code == 0
        assert "🎨 Color-Narrator - Narrate command" in result.output

    def test_narrate_command_with_debug(self, cli_runner, temp_dirs):
        """Test narrate command with debug flag."""
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                "--debug",
            ],
        )

        assert result.exit_code == 0
        assert "🎨 Color-Narrator - Narrate command" in result.output

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

    def test_narrate_command_no_arguments(self, cli_runner):
        """Test narrate command with no arguments (expects failure without valid defaults)."""
        result = cli_runner.invoke(app, ["narrate"])

        assert result.exit_code == 1  # Expected to fail without valid paths
        assert "🎨 Color-Narrator - Narrate command" in result.output

    def test_validate_command_no_arguments(self, cli_runner):
        """Test validate command with no arguments (expects failure without valid defaults)."""
        result = cli_runner.invoke(app, ["validate"])

        assert result.exit_code == 1  # Expected to fail without valid paths
        assert "🔍 Color-Narrator - Validate command" in result.output

    def test_narrate_command_short_flags(self, cli_runner, temp_dirs):
        """Test narrate command with short flag variants."""
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "-i",
                str(temp_dirs["images_dir"]),
                "-o",
                str(temp_dirs["overlays_dir"]),
                "-j",
                str(temp_dirs["mono_jsonl"]),
                "-b",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "🎨 Color-Narrator - Narrate command" in result.output

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

    def test_narrate_command_pathlib_conversion(self, cli_runner, temp_dirs):
        """Test that string paths are properly converted to Path objects."""
        # This tests that typer properly handles Path type annotations
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ],
        )

        # Should not raise any type errors
        assert result.exit_code == 0

    def test_app_direct_execution(self):
        """Test app can be executed directly."""
        # Test that the app can be called (this simulates python -m ... execution)
        assert app is not None
        assert isinstance(app, typer.Typer)

    def test_command_descriptions_present(self, cli_runner):
        """Test that commands have proper descriptions."""
        # Test narrate command has description
        result = cli_runner.invoke(app, ["narrate", "--help"])
        assert "Generate natural language color descriptions" in result.output

        # Test validate command has description
        result = cli_runner.invoke(app, ["validate", "--help"])
        assert "Validate existing color narrations" in result.output

    def test_command_examples_present(self, cli_runner):
        """Test that commands include usage examples."""
        result = cli_runner.invoke(app, ["narrate", "--help"])

        # Check for example usage
        assert "Example:" in result.output
        assert "imageworks-color-narrator narrate" in result.output

    @pytest.mark.parametrize("flag", ["--dry-run", "--debug"])
    def test_boolean_flags(self, cli_runner, temp_dirs, flag):
        """Test boolean flag handling."""
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
                flag,
            ],
        )

        assert result.exit_code == 0
        if flag == "--dry-run":
            assert "🔍 DRY RUN MODE - No files will be modified" in result.output

    def test_batch_size_validation(self, cli_runner, temp_dirs):
        """Test batch size parameter validation."""
        # Valid batch size
        result = cli_runner.invoke(
            app,
            [
                "narrate",
                "--batch-size",
                "4",
                "--images",
                str(temp_dirs["images_dir"]),
                "--overlays",
                str(temp_dirs["overlays_dir"]),
                "--mono-jsonl",
                str(temp_dirs["mono_jsonl"]),
            ],
        )
        assert result.exit_code == 0

        # Invalid batch size (should be handled by typer)
        result = cli_runner.invoke(app, ["narrate", "--batch-size", "invalid"])
        assert result.exit_code != 0

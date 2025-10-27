"""
Unit tests for VLMMonoInterpreter with backend client integration.
Tests Phase 1 changes: proxy routing and backend client usage.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from imageworks.apps.color_narrator.core.vlm_mono_interpreter import (
    VLMMonoInterpreter,
    VLMMonoResult,
)


class TestVLMMonoInterpreterProxyIntegration:
    """Test VLMMonoInterpreter uses backend client and routes through proxy."""

    def test_default_base_url_is_proxy(self):
        """Verify default base_url points to chat proxy (port 8100)."""
        interpreter = VLMMonoInterpreter()
        assert interpreter.base_url == "http://localhost:8100/v1"

    def test_default_model_is_registry_name(self):
        """Verify default model uses registry naming convention."""
        interpreter = VLMMonoInterpreter()
        assert interpreter.model == "qwen3-vl-8b-instruct_(FP8)"

    def test_custom_base_url_accepted(self):
        """Verify custom base_url can be provided."""
        interpreter = VLMMonoInterpreter(base_url="http://example.com:8080/v1")
        assert interpreter.base_url == "http://example.com:8080/v1"

    @patch(
        "imageworks.apps.color_narrator.core.vlm_mono_interpreter.create_backend_client"
    )
    @patch("imageworks.apps.color_narrator.core.vlm_mono_interpreter.Image.open")
    def test_uses_backend_client(self, mock_image_open, mock_create_client):
        """Verify interpret_mono_data uses create_backend_client instead of requests."""
        # Mock image
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = (512, 512)
        mock_img.save = Mock()
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock backend client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="VERDICT: monochrome\nTECHNICAL: Test\nVISUAL: Test\nSUMMARY: Test"
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_create_client.return_value = mock_client

        # Execute
        interpreter = VLMMonoInterpreter()
        mono_data = {
            "method": "hsv",
            "mode": "RGB",
            "dominant_color": "gray",
            "dominant_hue_deg": 0,
            "top_colors": ["#808080"],
            "top_weights": [1.0],
            "colorfulness": 0.5,
            "chroma_max": 10,
            "chroma_p95": 8,
            "chroma_p99": 9,
            "hue_std_deg": 2,
            "hue_concentration": 0.9,
            "hue_bimodality": 0.1,
            "mean_hue_highs_deg": 5,
            "mean_hue_shadows_deg": 4,
            "delta_h_highs_shadows_deg": 1,
            "channel_max_diff": 5,
            "sat_median": 0.1,
        }

        result = interpreter.interpret_mono_data(mono_data, Path("test.jpg"))

        # Verify backend client was created with correct parameters
        mock_create_client.assert_called_once_with(
            base_url="http://localhost:8100/v1",
            api_key="EMPTY",
            timeout=120,
        )

        # Verify client was used (not requests.post)
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "qwen3-vl-8b-instruct_(FP8)"
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["stream"] is False

        # Verify result
        assert isinstance(result, VLMMonoResult)
        assert result.verdict == "monochrome"
        assert result.vlm_model == "qwen3-vl-8b-instruct_(FP8)"

    @patch(
        "imageworks.apps.color_narrator.core.vlm_mono_interpreter.create_backend_client"
    )
    @patch("imageworks.apps.color_narrator.core.vlm_mono_interpreter.Image.open")
    def test_error_handling(self, mock_image_open, mock_create_client):
        """Verify proper error handling when backend client fails."""
        # Mock image
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = (512, 512)
        mock_img.save = Mock()
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock backend client to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_create_client.return_value = mock_client

        # Execute and verify exception is raised
        interpreter = VLMMonoInterpreter()
        mono_data = {
            "method": "hsv",
            "mode": "RGB",
            "dominant_color": "gray",
            "dominant_hue_deg": 0,
            "top_colors": ["#808080"],
            "top_weights": [1.0],
            "colorfulness": 0.5,
            "chroma_max": 10,
            "chroma_p95": 8,
            "chroma_p99": 9,
            "hue_std_deg": 2,
            "hue_concentration": 0.9,
            "hue_bimodality": 0.1,
            "mean_hue_highs_deg": 5,
            "mean_hue_shadows_deg": 4,
            "delta_h_highs_shadows_deg": 1,
            "channel_max_diff": 5,
            "sat_median": 0.1,
        }

        with pytest.raises(Exception) as exc_info:
            interpreter.interpret_mono_data(mono_data, Path("test.jpg"))

        assert "API Error" in str(exc_info.value)

    def test_parameters_passed_correctly(self):
        """Verify custom parameters are stored correctly."""
        interpreter = VLMMonoInterpreter(
            base_url="http://custom:9000/v1",
            model="custom-model",
            timeout=60,
            max_tokens=1000,
            temperature=0.5,
        )

        assert interpreter.base_url == "http://custom:9000/v1"
        assert interpreter.model == "custom-model"
        assert interpreter.timeout == 60
        assert interpreter.max_tokens == 1000
        assert interpreter.temperature == 0.5


class TestVLMMonoInterpreterResponseParsing:
    """Test response parsing logic."""

    def test_parse_vlm_response_complete(self):
        """Test parsing a complete VLM response."""
        interpreter = VLMMonoInterpreter()
        content = """
VERDICT: monochrome
TECHNICAL: The image shows very low colorfulness and high hue concentration.
VISUAL: The image appears predominantly gray with minimal color variation.
SUMMARY: This is a true monochrome image with no significant color content.
"""
        verdict, technical, visual, summary = interpreter._parse_vlm_response(content)

        assert verdict == "monochrome"
        assert "low colorfulness" in technical
        assert "predominantly gray" in visual
        assert "true monochrome" in summary

    def test_parse_vlm_response_multiline(self):
        """Test parsing a multi-line VLM response."""
        interpreter = VLMMonoInterpreter()
        content = """
VERDICT: colored
TECHNICAL: The image shows moderate colorfulness.
Additional technical details spanning multiple lines.
VISUAL: The image contains visible color variations.
More visual description here.
SUMMARY: This is a colored image.
"""
        verdict, technical, visual, summary = interpreter._parse_vlm_response(content)

        assert verdict == "colored"
        assert "moderate colorfulness" in technical
        assert "multiple lines" in technical
        assert "color variations" in visual
        assert "colored image" in summary

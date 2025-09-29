"""Tests for VLM inference client and response handling.

Tests VLM client initialization, request formatting, response parsing,
and error handling for Qwen2-VL-2B inference.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import base64

from imageworks.apps.color_narrator.core.vlm import (
    VLMBackend,
    VLMClient,
    VLMRequest,
    VLMResponse,
)


class TestVLMClient:
    """Test cases for VLM client functionality."""

    def test_client_initialization(self):
        """Test VLM client initializes with correct defaults."""
        client = VLMClient()

        assert client.base_url == "http://localhost:8000/v1"
        assert client.model_name == "Qwen2-VL-2B-Instruct"
        assert client.api_key == "EMPTY"
        assert client.timeout == 120
        assert client.backend == VLMBackend.VLLM

    def test_client_custom_config(self):
        """Test VLM client initializes with custom configuration."""
        client = VLMClient(
            base_url="http://custom:9000/v1",
            model_name="custom/model",
            api_key="test-key",
            timeout=60,
            backend="lmdeploy",
        )

        assert client.base_url == "http://custom:9000/v1"
        assert client.model_name == "custom/model"
        assert client.api_key == "test-key"
        assert client.timeout == 60
        assert client.backend == VLMBackend.LMDEPLOY

    def test_client_triton_backend_health(self):
        """TensorRT/Triton stub backend reports helpful health error."""
        client = VLMClient(backend=VLMBackend.TRITON)

        assert client.health_check() is False
        assert client.last_error is not None
        assert "TensorRT-LLM backend" in client.last_error

    @patch("builtins.open", create=True)
    def test_encode_image_success(self, mock_open):
        """Test successful image encoding to base64."""
        # Mock file content
        mock_file_content = b"fake_image_data"
        mock_open.return_value.__enter__.return_value.read.return_value = (
            mock_file_content
        )

        client = VLMClient()
        result = client.encode_image(Path("/fake/image.jpg"))

        expected = base64.b64encode(mock_file_content).decode("utf-8")
        assert result == expected

    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_encode_image_failure(self, mock_open):
        """Test image encoding failure handling."""
        client = VLMClient()

        with pytest.raises(RuntimeError, match="Failed to encode image"):
            client.encode_image(Path("/nonexistent/image.jpg"))

    def test_create_color_narration_prompt_default(self):
        """Test default prompt template creation."""
        client = VLMClient()
        mono_data = {
            "hue_analysis": "Warm cast detected",
            "chroma_analysis": "Low chroma levels",
            "contamination_level": 0.3,
        }

        prompt = client.create_color_narration_prompt(mono_data)

        assert "Warm cast detected" in prompt
        assert "Low chroma levels" in prompt
        assert "contamination_level" in prompt or "0.3" in prompt

    def test_create_color_narration_prompt_custom(self):
        """Test custom prompt template creation."""
        client = VLMClient()
        mono_data = {"test_key": "test_value"}
        template = "Custom template with {test_key}"

        prompt = client.create_color_narration_prompt(mono_data, template)

        assert prompt == "Custom template with test_value"

    @patch.object(VLMClient, "encode_image")
    @patch("requests.Session.post")
    def test_infer_single_success(self, mock_post, mock_encode):
        """Test successful single image inference."""
        # Setup mocks
        mock_encode.return_value = "fake_base64_data"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test color description"}}],
            "usage": {"total_time": 1.5},
        }
        mock_post.return_value = mock_response

        # Create test request
        client = VLMClient()
        request = VLMRequest(
            image_path=Path("/fake/image.jpg"),
            overlay_path=Path("/fake/overlay.png"),
            mono_data={"contamination_level": 0.5},
            prompt_template="Test prompt",
        )

        # Execute inference
        result = client.infer_single(request)

        # Verify results
        assert isinstance(result, VLMResponse)
        assert result.description == "Test color description"
        assert result.error is None
        assert result.processing_time == 1.5

    @patch.object(VLMClient, "encode_image")
    @patch("requests.Session.post")
    def test_infer_single_api_error(self, mock_post, mock_encode):
        """Test handling of API error responses."""
        mock_encode.return_value = "fake_base64_data"
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = VLMClient()
        request = VLMRequest(
            image_path=Path("/fake/image.jpg"),
            overlay_path=Path("/fake/overlay.png"),
            mono_data={},
            prompt_template="Test prompt",
        )

        result = client.infer_single(request)

        assert result.error is not None
        assert "API error 500" in result.error
        assert result.description == ""
        assert result.confidence == 0.0

    @patch.object(VLMClient, "encode_image", return_value="stub")
    def test_triton_backend_infer_single(self, mock_encode):
        """Triton stub backend should surface informative error."""
        client = VLMClient(backend=VLMBackend.TRITON)
        request = VLMRequest(
            image_path=Path("/fake/image.jpg"),
            overlay_path=Path("/fake/overlay.png"),
            mono_data={},
            prompt_template="Test",
        )

        response = client.infer_single(request)

        assert response.error is not None
        assert "TensorRT-LLM backend" in response.error
        assert response.description == ""

    @patch.object(VLMClient, "encode_image", side_effect=Exception("Encoding failed"))
    def test_infer_single_exception(self, mock_encode):
        """Test handling of exceptions during inference."""
        client = VLMClient()
        request = VLMRequest(
            image_path=Path("/fake/image.jpg"),
            overlay_path=Path("/fake/overlay.png"),
            mono_data={},
            prompt_template="Test prompt",
        )

        result = client.infer_single(request)

        assert result.error is not None
        assert "Inference error" in result.error
        assert result.description == ""
        assert result.confidence == 0.0

    def test_extract_color_regions(self):
        """Test color region extraction from description text."""
        client = VLMClient()

        description = (
            "The background shows warm tones while the skin has cool highlights"
        )
        regions = client._extract_color_regions(description)

        assert "background" in regions
        assert "skin" in regions

    def test_estimate_confidence(self):
        """Test confidence estimation from description characteristics."""
        client = VLMClient()

        # High confidence indicators
        high_conf = client._estimate_confidence("The clear color cast is obvious")
        assert high_conf == 0.9

        # Medium confidence indicators
        med_conf = client._estimate_confidence("There are subtle color traces")
        assert med_conf == 0.7

        # Low confidence indicators
        low_conf = client._estimate_confidence("Minimal color contamination detected")
        assert low_conf == 0.5

        # Default confidence
        default_conf = client._estimate_confidence("Some color description")
        assert default_conf == 0.8

    @patch("requests.Session.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = VLMClient()
        assert client.health_check() is True

    @patch("requests.Session.get")
    def test_health_check_failure(self, mock_get):
        """Test failed health check."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        client = VLMClient()
        assert client.health_check() is False

    @patch("requests.Session.get", side_effect=Exception("Connection failed"))
    def test_health_check_exception(self, mock_get):
        """Test health check with connection exception."""
        client = VLMClient()
        assert client.health_check() is False


class TestVLMDataClasses:
    """Test cases for VLM data classes."""

    def test_vlm_response_creation(self):
        """Test VLMResponse dataclass creation."""
        response = VLMResponse(
            description="Test description",
            confidence=0.85,
            color_regions=["background", "skin"],
            processing_time=1.2,
            error=None,
        )

        assert response.description == "Test description"
        assert response.confidence == 0.85
        assert response.color_regions == ["background", "skin"]
        assert response.processing_time == 1.2
        assert response.error is None

    def test_vlm_request_creation(self):
        """Test VLMRequest dataclass creation."""
        request = VLMRequest(
            image_path=Path("/test/image.jpg"),
            overlay_path=Path("/test/overlay.png"),
            mono_data={"contamination": 0.3},
            prompt_template="Test prompt: {contamination}",
        )

        assert request.image_path == Path("/test/image.jpg")
        assert request.overlay_path == Path("/test/overlay.png")
        assert request.mono_data == {"contamination": 0.3}
        assert request.prompt_template == "Test prompt: {contamination}"

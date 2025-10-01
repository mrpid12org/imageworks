"""
Tests for format detection functionality.
"""

from imageworks.tools.model_downloader.formats import (
    FormatDetector,
    FormatInfo,
    detect_format,
    get_primary_format,
)


class TestFormatDetector:
    """Test the FormatDetector class."""

    def test_init(self):
        """Test FormatDetector initialization."""
        detector = FormatDetector()
        assert detector is not None
        assert hasattr(detector, "compiled_patterns")

    def test_detect_from_filename_gguf(self):
        """Test GGUF detection from filename."""
        detector = FormatDetector()

        # Test various GGUF patterns
        gguf_files = [
            "model.gguf",
            "llama-7b-chat.Q4_K_M.gguf",
            "mistral-instruct-q4_0.gguf",
            "model-f16.gguf",
        ]

        for filename in gguf_files:
            formats = detector.detect_from_filename(filename)
            gguf_formats = [f for f in formats if f.format_type == "gguf"]
            assert len(gguf_formats) >= 1, f"Failed to detect GGUF in {filename}"
            assert gguf_formats[0].confidence >= 0.7

    def test_detect_from_filename_awq(self):
        """Test AWQ detection from filename."""
        detector = FormatDetector()

        awq_files = [
            "model-awq.bin",
            "llama-7b-awq-4bit",
            "awq_model.safetensors",
            "model_autoawq.bin",
        ]

        for filename in awq_files:
            formats = detector.detect_from_filename(filename)
            awq_formats = [f for f in formats if f.format_type == "awq"]
            assert len(awq_formats) >= 1, f"Failed to detect AWQ in {filename}"

    def test_detect_from_filename_safetensors(self):
        """Test Safetensors detection from filename."""
        detector = FormatDetector()

        safetensors_files = [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "pytorch_model.safetensors",
        ]

        for filename in safetensors_files:
            formats = detector.detect_from_filename(filename)
            safetensors_formats = [f for f in formats if f.format_type == "safetensors"]
            assert (
                len(safetensors_formats) >= 1
            ), f"Failed to detect safetensors in {filename}"
            assert safetensors_formats[0].confidence >= 0.8

    def test_detect_from_filelist(self):
        """Test detection from multiple files."""
        detector = FormatDetector()

        # Mixed file list with different formats
        filenames = [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer.json",
            "README.md",
        ]

        formats = detector.detect_from_filelist(filenames)

        # Should detect safetensors as primary format
        safetensors_formats = [f for f in formats if f.format_type == "safetensors"]
        assert len(safetensors_formats) == 1
        assert safetensors_formats[0].confidence > 0.8

    def test_detect_from_config_awq(self, sample_awq_config_json):
        """Test AWQ detection from config.json."""
        detector = FormatDetector()

        formats = detector.detect_from_config(sample_awq_config_json)

        awq_formats = [f for f in formats if f.format_type == "awq"]
        assert len(awq_formats) == 1
        assert awq_formats[0].confidence >= 0.9
        assert awq_formats[0].quantization_details is not None
        assert awq_formats[0].quantization_details["quant_method"] == "awq"

    def test_detect_from_config_invalid_json(self):
        """Test handling of invalid JSON config."""
        detector = FormatDetector()

        invalid_json = "{ invalid json }"
        formats = detector.detect_from_config(invalid_json)

        assert formats == []

    def test_detect_from_model_name(self):
        """Test format detection from model names."""
        detector = FormatDetector()

        test_cases = [
            ("microsoft/DialoGPT-medium-awq", "awq"),
            ("TheBloke/Llama-2-7B-Chat-GGUF", "gguf"),
            ("casperhansen/llama-7b-instruct-awq", "awq"),
            ("huggingface/gptq-model", "gptq"),
        ]

        for model_name, expected_format in test_cases:
            formats = detector.detect_from_model_name(model_name)
            format_types = [f.format_type for f in formats]
            assert (
                expected_format in format_types
            ), f"Failed to detect {expected_format} in {model_name}"

    def test_detect_comprehensive(self):
        """Test comprehensive detection combining all sources."""
        detector = FormatDetector()

        model_name = "microsoft/DialoGPT-medium-awq"
        filenames = ["config.json", "model.safetensors", "tokenizer.json"]
        config_content = '{"quantization_config": {"quant_method": "awq"}}'

        formats = detector.detect_comprehensive(
            model_name=model_name, filenames=filenames, config_content=config_content
        )

        # Should detect both AWQ and safetensors
        format_types = [f.format_type for f in formats]
        assert "awq" in format_types
        assert "safetensors" in format_types

        # AWQ should have higher confidence due to config
        awq_format = next(f for f in formats if f.format_type == "awq")
        safetensors_format = next(f for f in formats if f.format_type == "safetensors")
        assert awq_format.confidence >= safetensors_format.confidence

    def test_get_primary_format(self):
        """Test getting primary format."""
        detector = FormatDetector()

        # Clear GGUF case
        primary = detector.get_primary_format(filenames=["model.gguf", "config.json"])

        assert primary is not None
        assert primary.format_type == "gguf"
        assert primary.confidence >= 0.8

    def test_get_primary_format_low_confidence(self):
        """Test primary format with low confidence threshold."""
        detector = FormatDetector()

        # Ambiguous case with high threshold
        primary = detector.get_primary_format(
            model_name="some-model", confidence_threshold=0.95
        )

        # Should return None due to low confidence
        assert primary is None or primary.confidence >= 0.95

    def test_is_format_type(self):
        """Test format type checking."""
        detector = FormatDetector()

        # Test GGUF detection
        is_gguf = detector.is_format_type("gguf", filenames=["model.Q4_K_M.gguf"])
        assert is_gguf is True

        # Test non-GGUF
        is_gguf = detector.is_format_type("gguf", filenames=["model.safetensors"])
        assert is_gguf is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_format(self):
        """Test detect_format convenience function."""
        formats = detect_format(filenames=["model.safetensors"])

        assert len(formats) >= 1
        safetensors_formats = [f for f in formats if f.format_type == "safetensors"]
        assert len(safetensors_formats) >= 1

    def test_get_primary_format(self):
        """Test get_primary_format convenience function."""
        primary = get_primary_format(filenames=["model.gguf"])

        assert primary == "gguf"

    def test_get_primary_format_none(self):
        """Test get_primary_format returns None for unclear cases."""
        primary = get_primary_format(filenames=["unknown.txt"])

        # Should return None or a low-confidence guess
        assert primary is None or primary in [
            "safetensors",
            "pytorch",
        ]  # Default fallbacks


class TestFormatInfo:
    """Test FormatInfo dataclass."""

    def test_format_info_creation(self):
        """Test FormatInfo creation."""
        format_info = FormatInfo(
            format_type="gguf",
            confidence=0.95,
            evidence=["file_extension: .gguf"],
            quantization_details={"bits": 4},
        )

        assert format_info.format_type == "gguf"
        assert format_info.confidence == 0.95
        assert format_info.evidence == ["file_extension: .gguf"]
        assert format_info.quantization_details == {"bits": 4}

    def test_format_info_defaults(self):
        """Test FormatInfo default values."""
        format_info = FormatInfo(format_type="awq", confidence=0.8, evidence=["config"])

        assert format_info.quantization_details is None

"""
Format detection and classification for AI models.

Automatically detects model formats (GGUF, AWQ, GPTQ, etc.) from various sources
including filenames, model configs, and repository metadata.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class FormatInfo:
    """Information about a detected model format."""

    format_type: str
    confidence: float
    evidence: List[str]
    quantization_details: Optional[Dict[str, Any]] = None


class FormatDetector:
    """Detects and classifies AI model formats."""

    # File extension patterns
    EXTENSION_PATTERNS = {
        "gguf": [r"\.gguf$"],
        "safetensors": [r"\.safetensors$"],
        "pytorch": [r"\.bin$", r"\.pth$", r"\.pt$"],
        "onnx": [r"\.onnx$"],
        "tensorflow": [r"\.pb$", r"saved_model\.pb$"],
    }

    # Model name patterns for quantization detection
    NAME_PATTERNS = {
        "awq": [
            r"[-_]awq[-_]",
            r"[-_]awq$",
            r"[-_]awq\.",  # For model-awq.bin
            r"^awq[-_]",
            r"autoawq",
            r"awq[-_]",  # Added pattern for awq_model, model-awq etc
        ],
        "gptq": [
            r"[-_]gptq[-_]",
            r"[-_]gptq$",
            r"[-_]gptq\.",  # For model-gptq.bin
            r"^gptq[-_]",
            r"autogptq",
            r"gptq[-_]",  # Added pattern for gptq-model etc
        ],
        "gguf": [
            r"[-_]gguf[-_]",
            r"[-_]gguf$",
            r"\.Q\d+_[KM]",  # GGUF quantization patterns like Q4_K_M
            r"[-_]q\d+[-_]",
            r"f16|f32|q2|q3|q4|q5|q6|q8",
        ],
        "bnb": [r"[-_]4bit[-_]", r"[-_]8bit[-_]", r"bitsandbytes"],
    }

    # Config.json quantization method mappings
    CONFIG_QUANTIZATION_METHODS = {
        "awq": ["awq"],
        "gptq": ["gptq", "autogptq"],
        "bnb": ["bitsandbytes", "bnb"],
        "eetq": ["eetq"],
        "hqq": ["hqq"],
    }

    def __init__(self):
        self.compiled_patterns = {}
        for format_type, patterns in self.NAME_PATTERNS.items():
            self.compiled_patterns[format_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def detect_from_filename(self, filename: str) -> List[FormatInfo]:
        """Detect format from a single filename."""
        formats = []
        filename_lower = filename.lower()

        # Check file extensions first
        for format_type, patterns in self.EXTENSION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename_lower):
                    formats.append(
                        FormatInfo(
                            format_type=format_type,
                            confidence=0.9,
                            evidence=[f"file_extension: {pattern}"],
                        )
                    )

        # Check name patterns for quantization
        for format_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(filename_lower):
                    formats.append(
                        FormatInfo(
                            format_type=format_type,
                            confidence=0.7,
                            evidence=[f"name_pattern: {pattern.pattern}"],
                        )
                    )

        return formats

    def detect_from_filelist(self, filenames: List[str]) -> List[FormatInfo]:
        """Detect format from a list of files."""
        all_formats = []
        format_counts = {}

        for filename in filenames:
            file_formats = self.detect_from_filename(filename)
            for format_info in file_formats:
                format_type = format_info.format_type
                if format_type not in format_counts:
                    format_counts[format_type] = {
                        "count": 0,
                        "evidence": [],
                        "total_confidence": 0.0,
                    }
                format_counts[format_type]["count"] += 1
                format_counts[format_type]["evidence"].extend(format_info.evidence)
                format_counts[format_type]["total_confidence"] += format_info.confidence

        # Convert counts to format info
        for format_type, data in format_counts.items():
            avg_confidence = data["total_confidence"] / data["count"]
            # Boost confidence for multiple files of same format
            if data["count"] > 1:
                avg_confidence = min(0.95, avg_confidence + 0.1)

            all_formats.append(
                FormatInfo(
                    format_type=format_type,
                    confidence=avg_confidence,
                    evidence=list(set(data["evidence"])),  # Remove duplicates
                )
            )

        # Sort by confidence
        return sorted(all_formats, key=lambda x: x.confidence, reverse=True)

    def detect_from_config(self, config_content: str) -> List[FormatInfo]:
        """Detect format from config.json content."""
        formats = []

        try:
            config = json.loads(config_content)
        except json.JSONDecodeError:
            return formats

        # Check for quantization_config
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            quant_method = quant_config.get("quant_method", "").lower()

            for format_type, methods in self.CONFIG_QUANTIZATION_METHODS.items():
                if quant_method in methods:
                    formats.append(
                        FormatInfo(
                            format_type=format_type,
                            confidence=0.95,
                            evidence=[f"config_quant_method: {quant_method}"],
                            quantization_details=quant_config,
                        )
                    )

        # Check model type and architecture for additional hints
        architectures = config.get("architectures", [])

        # Some architectures are more commonly found in certain formats
        if any("llama" in arch.lower() for arch in architectures):
            # LLaMA models often have GGUF variants
            pass  # Don't auto-assume format just from architecture

        return formats

    def detect_from_model_name(self, model_name: str) -> List[FormatInfo]:
        """Detect format from full model name (owner/repo)."""
        formats = []
        name_lower = model_name.lower()

        for format_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(name_lower):
                    formats.append(
                        FormatInfo(
                            format_type=format_type,
                            confidence=0.6,  # Lower confidence for name alone
                            evidence=[f"model_name_pattern: {pattern.pattern}"],
                        )
                    )

        return formats

    def detect_comprehensive(
        self,
        model_name: Optional[str] = None,
        filenames: Optional[List[str]] = None,
        config_content: Optional[str] = None,
    ) -> List[FormatInfo]:
        """Comprehensive format detection using all available information."""
        all_formats = []

        if model_name:
            all_formats.extend(self.detect_from_model_name(model_name))

        if filenames:
            all_formats.extend(self.detect_from_filelist(filenames))

        if config_content:
            all_formats.extend(self.detect_from_config(config_content))

        # Merge formats of the same type
        merged_formats = {}
        for format_info in all_formats:
            format_type = format_info.format_type
            if format_type not in merged_formats:
                merged_formats[format_type] = format_info
            else:
                # Merge evidence and take highest confidence
                existing = merged_formats[format_type]
                existing.confidence = max(existing.confidence, format_info.confidence)
                existing.evidence.extend(format_info.evidence)
                existing.evidence = list(set(existing.evidence))  # Remove duplicates

                # Merge quantization details
                if format_info.quantization_details:
                    existing.quantization_details = format_info.quantization_details

        # Convert back to list and sort by confidence
        result = list(merged_formats.values())
        return sorted(result, key=lambda x: x.confidence, reverse=True)

    def get_primary_format(
        self,
        model_name: Optional[str] = None,
        filenames: Optional[List[str]] = None,
        config_content: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> Optional[FormatInfo]:
        """Get the primary (most confident) format detection result."""
        formats = self.detect_comprehensive(model_name, filenames, config_content)

        if formats and formats[0].confidence >= confidence_threshold:
            return formats[0]

        return None

    def is_format_type(
        self,
        format_type: str,
        model_name: Optional[str] = None,
        filenames: Optional[List[str]] = None,
        config_content: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ) -> bool:
        """Check if the model matches a specific format type."""
        formats = self.detect_comprehensive(model_name, filenames, config_content)

        for format_info in formats:
            if (
                format_info.format_type == format_type
                and format_info.confidence >= confidence_threshold
            ):
                return True

        return False


# Convenience functions
def detect_format(
    model_name: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    config_content: Optional[str] = None,
) -> List[FormatInfo]:
    """Convenience function for format detection."""
    detector = FormatDetector()
    return detector.detect_comprehensive(model_name, filenames, config_content)


def get_primary_format(
    model_name: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    config_content: Optional[str] = None,
) -> Optional[str]:
    """Get the primary format type as a string."""
    detector = FormatDetector()
    primary = detector.get_primary_format(model_name, filenames, config_content)
    return primary.format_type if primary else None

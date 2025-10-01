"""
URL analysis and HuggingFace repository inspection.

Handles parsing of HuggingFace URLs, repository analysis, and file discovery
for intelligent model downloading.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import requests
from .formats import FormatDetector, FormatInfo


@dataclass
class FileInfo:
    """Information about a file in a repository."""

    path: str
    size: int
    sha: Optional[str] = None
    lfs: bool = False


@dataclass
class RepositoryInfo:
    """Information about a HuggingFace repository."""

    owner: str
    repo: str
    branch: str = "main"
    model_type: Optional[str] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of repository analysis."""

    repository: RepositoryInfo
    files: Dict[str, List[FileInfo]]  # categorized files
    formats: List[FormatInfo]
    total_size: int
    config_content: Optional[str] = None


class URLAnalyzer:
    """Analyzes URLs and HuggingFace repositories."""

    URL_PATTERNS = {
        "huggingface_model": re.compile(
            r"https://huggingface\.co/([^/]+)/([^/]+)/?(?:\?.*)?$"
        ),
        "huggingface_file": re.compile(
            r"https://huggingface\.co/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)$"
        ),
        "huggingface_tree": re.compile(
            r"https://huggingface\.co/([^/]+)/([^/]+)/tree/([^/]+)/?(?:\?.*)?$"
        ),
        "direct_file": re.compile(
            r"https?://.*\.(safetensors|gguf|bin|pth|onnx|pt)(?:\?.*)?$"
        ),
        "huggingface_shorthand": re.compile(
            r"^([\w.-]+)/([\w.-]+?)(?:@([\w./-]+))?$"
        ),
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.format_detector = FormatDetector()

    def parse_url(self, url: str) -> Tuple[str, Optional[Tuple]]:
        """Parse a URL and return its type and components."""
        url_clean = url.split("#")[0]  # Remove fragments

        for pattern_name, pattern in self.URL_PATTERNS.items():
            match = pattern.match(url_clean)
            if match:
                return pattern_name, match.groups()

        return "unknown", None

    def analyze_huggingface_repo(
        self, owner: str, repo: str, branch: str = "main"
    ) -> AnalysisResult:
        """Analyze a HuggingFace repository comprehensively."""

        # Get model info
        model_info = self._get_model_info(owner, repo)

        # Get repository info
        repo_info = RepositoryInfo(
            owner=owner,
            repo=repo,
            branch=branch,
            model_type=model_info.get("pipeline_tag"),
            pipeline_tag=model_info.get("pipeline_tag"),
            library_name=model_info.get("library_name"),
            tags=model_info.get("tags", []),
        )

        # Get file list
        files_info = self._get_repo_files(owner, repo, branch)
        categorized_files = self._categorize_files(files_info)

        # Get config content for format detection
        config_content = self._get_config_content(owner, repo, branch)

        # Detect formats
        filenames = [f.path for f in files_info]
        formats = self.format_detector.detect_comprehensive(
            model_name=f"{owner}/{repo}",
            filenames=filenames,
            config_content=config_content,
        )

        # Calculate total size
        total_size = sum(f.size for f in files_info)

        return AnalysisResult(
            repository=repo_info,
            files=categorized_files,
            formats=formats,
            total_size=total_size,
            config_content=config_content,
        )

    def analyze_url(self, url: str) -> AnalysisResult:
        """Analyze any supported URL."""
        pattern_type, groups = self.parse_url(url)

        if pattern_type == "huggingface_model":
            owner, repo = groups
            return self.analyze_huggingface_repo(owner, repo)

        elif pattern_type == "huggingface_tree":
            owner, repo, branch = groups
            return self.analyze_huggingface_repo(owner, repo, branch)

        elif pattern_type == "huggingface_file":
            owner, repo, branch, filepath = groups
            # For single file, analyze the whole repo but highlight the specific file
            result = self.analyze_huggingface_repo(owner, repo, branch)
            # Mark the specific file as priority
            for category_files in result.files.values():
                for file_info in category_files:
                    if file_info.path == filepath:
                        file_info.priority = True
            return result

        elif pattern_type == "huggingface_shorthand":
            owner, repo, branch = groups
            return self.analyze_huggingface_repo(
                owner,
                repo,
                branch or "main",
            )

        else:
            # Support bare owner/repo identifiers even if they do not
            # strictly match the shorthand regex (e.g. extra whitespace).
            normalized = url.strip()
            if normalized and "/" in normalized and not normalized.startswith(
                ("http://", "https://")
            ):
                owner, repo_branch = normalized.split("/", 1)
                repo, _, branch = repo_branch.partition("@")
                if owner and repo:
                    return self.analyze_huggingface_repo(
                        owner,
                        repo,
                        branch or "main",
                    )
            raise ValueError(f"Unsupported URL type: {pattern_type}")

    def _get_model_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get model information from HuggingFace API."""
        api_url = f"https://huggingface.co/api/models/{owner}/{repo}"

        try:
            response = requests.get(api_url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Could not fetch model info: {e}")
            return {}

    def _get_repo_files(
        self, owner: str, repo: str, branch: str = "main"
    ) -> List[FileInfo]:
        """Get file list from HuggingFace repository."""
        api_url = f"https://huggingface.co/api/models/{owner}/{repo}/tree/{branch}"

        try:
            response = requests.get(api_url, timeout=self.timeout)
            response.raise_for_status()
            files_data = response.json()

            files = []
            for file_data in files_data:
                if file_data["type"] == "file":
                    files.append(
                        FileInfo(
                            path=file_data["path"],
                            size=file_data.get("size", 0),
                            sha=file_data.get("oid"),
                            lfs=file_data.get("lfs", {}).get("size") is not None,
                        )
                    )

            return files

        except requests.RequestException as e:
            print(f"Warning: Could not fetch file list: {e}")
            return []

    def _get_config_content(
        self, owner: str, repo: str, branch: str = "main"
    ) -> Optional[str]:
        """Get config.json content from repository."""
        config_url = f"https://huggingface.co/{owner}/{repo}/raw/{branch}/config.json"

        try:
            response = requests.get(config_url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def _categorize_files(self, files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        """Categorize files by their purpose and importance."""
        categories = {
            "model_weights": [],  # .safetensors, .bin, .gguf, .pth
            "config": [],  # config files
            "tokenizer": [],  # tokenizer files
            "optional": [],  # documentation, examples
            "large_optional": [],  # large optional files
            "ignore": [],  # files to ignore
        }

        for file_info in files:
            path = file_info.path.lower()
            size = file_info.size

            # Model weights (primary files)
            if any(
                path.endswith(ext)
                for ext in [".safetensors", ".bin", ".gguf", ".pth", ".pt"]
            ):
                categories["model_weights"].append(file_info)

            # Configuration files (essential)
            elif any(
                name in path
                for name in [
                    "config.json",
                    "generation_config.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ]
            ):
                categories["config"].append(file_info)

            # Tokenizer files (essential for text models)
            elif any(
                name in path
                for name in [
                    "tokenizer.json",
                    "tokenizer.model",
                    "vocab.txt",
                    "merges.txt",
                    "added_tokens.json",
                ]
            ):
                categories["tokenizer"].append(file_info)

            # Small optional files
            elif (
                any(
                    path.endswith(ext)
                    for ext in [".md", ".txt", ".license", ".gitattributes"]
                )
                or size < 1024 * 1024
            ):  # < 1MB
                categories["optional"].append(file_info)

            # Large optional files
            elif size > 100 * 1024 * 1024:  # > 100MB
                # Could be model variants, training data, etc.
                categories["large_optional"].append(file_info)

            # Ignore patterns
            elif any(
                pattern in path for pattern in [".git", "__pycache__", ".ds_store"]
            ):
                categories["ignore"].append(file_info)

            else:
                # Default to optional for unknown small files
                if size < 10 * 1024 * 1024:  # < 10MB
                    categories["optional"].append(file_info)
                else:
                    categories["large_optional"].append(file_info)

        return categories

    def create_download_plan(
        self,
        analysis: AnalysisResult,
        format_preference: Optional[str] = None,
        include_optional: bool = False,
    ) -> Dict[str, Any]:
        """Create a download plan based on analysis results."""

        # Determine target format
        if format_preference:
            target_formats = [
                f for f in analysis.formats if f.format_type == format_preference
            ]
            if not target_formats:
                raise ValueError(
                    f"Requested format '{format_preference}' not available"
                )
            target_format = target_formats[0]
        else:
            # Use highest confidence format
            target_format = analysis.formats[0] if analysis.formats else None

        # Required files
        required_files = []
        required_files.extend(analysis.files.get("config", []))
        required_files.extend(analysis.files.get("tokenizer", []))

        # Model weights - filter by format if needed
        model_weights = analysis.files.get("model_weights", [])
        if target_format:
            if target_format.format_type == "gguf":
                model_weights = [
                    f for f in model_weights if f.path.lower().endswith(".gguf")
                ]
            elif target_format.format_type == "safetensors":
                model_weights = [
                    f for f in model_weights if f.path.lower().endswith(".safetensors")
                ]
            # Add more format-specific filtering as needed

        required_files.extend(model_weights)

        # Optional files
        optional_files = []
        if include_optional:
            optional_files.extend(analysis.files.get("optional", []))

        # Calculate sizes
        required_size = sum(f.size for f in required_files)
        optional_size = sum(f.size for f in optional_files)

        return {
            "repository": analysis.repository,
            "target_format": target_format.format_type if target_format else "unknown",
            "required_files": required_files,
            "optional_files": optional_files,
            "required_size": required_size,
            "optional_size": optional_size,
            "total_size": required_size + optional_size,
            "format_details": target_format,
        }


def analyze_url(url: str) -> AnalysisResult:
    """Convenience function to analyze a URL."""
    analyzer = URLAnalyzer()
    return analyzer.analyze_url(url)


def create_download_plan(
    url: str, format_preference: Optional[str] = None, include_optional: bool = False
) -> Dict[str, Any]:
    """Convenience function to create a download plan from URL."""
    analyzer = URLAnalyzer()
    analysis = analyzer.analyze_url(url)
    return analyzer.create_download_plan(analysis, format_preference, include_optional)

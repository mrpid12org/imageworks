"""
Main model downloader implementation.

Coordinates URL analysis, format detection, directory routing, and download execution
using aria2c for optimal performance.
"""

import logging
import os
import re
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Union, Set
from dataclasses import dataclass

from .config import get_config, DownloaderConfig
from imageworks.model_loader.registry import load_registry, update_entries, remove_entry
from imageworks.model_loader.download_adapter import record_download
from imageworks.model_loader.models import RegistryEntry
from .url_analyzer import URLAnalyzer
from .formats import FormatDetector
from .format_utils import detect_format_and_quant
import hashlib

_README_HASH_LIMIT = 10 * 1024 * 1024  # 10 MiB safety guard
_README_PARSE_LIMIT = 512 * 1024  # parse at most first 512 KiB for regex hints


def _is_minimal_doc(path: str) -> bool:
    """Return True for README*/LICENSE* style documentation files."""

    name = Path(path).name.lower()
    if not name:
        return False
    if name.startswith("readme"):
        return True
    if name.startswith("license"):
        return True
    return False


def _select_primary_readme(root: Path) -> Optional[Path]:
    """Pick the highest-priority README within the downloaded repo."""

    candidates = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name.lower().startswith("readme")
    ]
    if not candidates:
        return None

    def _priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if name == "readme.md":
            rank = 0
        elif name.endswith(".md"):
            rank = 1
        elif name.endswith(".txt"):
            rank = 2
        elif "." not in name:
            rank = 3
        else:
            rank = 4
        try:
            depth = len(path.relative_to(root).parts)
        except Exception:  # noqa: BLE001
            depth = len(path.parts)
        return (rank, depth, path.as_posix())

    return sorted(candidates, key=_priority)[0]


def _hash_small_file(path: Path) -> Optional[str]:
    """Compute sha256 when the file is below the configured threshold."""

    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _README_HASH_LIMIT:
        return None
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _read_text_limited(path: Path, limit: int = _README_PARSE_LIMIT) -> str:
    try:
        with path.open("rb") as fh:
            data = fh.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="ignore")


_QUANT_SCHEME_PATTERN = re.compile(
    r"\b(awq|gptq|mxfp4|bnb|fp8|fp16|bf16|int4|int8|squeezellm|iq4_xs)\b",
    re.IGNORECASE,
)
_QUANT_DETAIL_WG_PATTERN = re.compile(r"\bw(\d+)\s*g(\d+)\b", re.IGNORECASE)
_QUANT_DETAIL_BIT_PATTERN = re.compile(r"\b(\d+)\s*[-/]?\s*bit\b", re.IGNORECASE)
_CONTAINER_PATTERN = re.compile(r"\b(gguf|safetensors)\b", re.IGNORECASE)
_BACKEND_PATTERN = re.compile(r"\b(vllm|llama\.cpp|ollama|lmdeploy)\b", re.IGNORECASE)
_TOOL_PATTERN = re.compile(r"\bfunction[- ]?call(?:ing)?\b|\btools?\b", re.IGNORECASE)
_REASONING_PATTERN = re.compile(r"\breasoning\b|\br1\b|\bo3\b|\bthink\b", re.IGNORECASE)
_LICENSE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"apache[- ]?2\.0", re.IGNORECASE), "apache-2.0"),
    (re.compile(r"mit\b", re.IGNORECASE), "mit"),
    (re.compile(r"llama\s*3", re.IGNORECASE), "llama-3"),
]


def _parse_readme_signals(text: str) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}

    schemes = {match.lower() for match in _QUANT_SCHEME_PATTERN.findall(text)}
    if schemes:
        signals["quant_schemes"] = sorted(schemes)

    details: set[str] = set()
    for bits, groups in _QUANT_DETAIL_WG_PATTERN.findall(text):
        details.add(f"w{bits}g{groups}")
    for phrase in _QUANT_DETAIL_BIT_PATTERN.findall(text):
        if phrase:
            normalized = f"{phrase.strip().replace(' ', '').replace('/', '')}".lower()
            # Ensure it ends with 'bit'
            if not normalized.endswith("bit"):
                normalized = f"{normalized}-bit"
            # Normalize `4bit` -> `4-bit`
            if normalized.endswith("bit") and not normalized.endswith("-bit"):
                normalized = normalized[:-3] + "-bit"
            details.add(normalized)
    if details:
        signals["quant_details"] = sorted(details)

    containers = {match.lower() for match in _CONTAINER_PATTERN.findall(text)}
    if containers:
        signals["containers"] = sorted(containers)

    backends = {match.lower() for match in _BACKEND_PATTERN.findall(text)}
    if backends:
        signals["backends"] = sorted(backends)

    if _TOOL_PATTERN.search(text):
        signals["tool_calls"] = True

    if _REASONING_PATTERN.search(text):
        signals["reasoning"] = True

    for pattern, label in _LICENSE_PATTERNS:
        if pattern.search(text):
            signals["license_hint"] = label
            break

    return signals


def _derive_producer(
    *,
    existing: Optional[str],
    hf_id: Optional[str],
    source_provider: Optional[str],
    served_model_id: Optional[str],
    download_path: Optional[str],
) -> Optional[str]:
    if existing:
        return existing.lower()
    if hf_id:
        owner = hf_id.split("/")[0]
        if owner:
            return owner.lower()
    if source_provider == "ollama" and served_model_id:
        normalized = (
            served_model_id.replace(":", "-").replace("/", "-").replace("\\", "-")
        )
        lowered = normalized.lower()
        if lowered.startswith("hf.co-"):
            tokens = [tok for tok in lowered.split("-") if tok]
            if len(tokens) >= 3:
                return tokens[1]
        if lowered.startswith("hf.co/"):
            tokens = [tok for tok in lowered.split("/") if tok]
            if len(tokens) >= 2:
                return tokens[1]
        tokens = [tok for tok in normalized.split("-") if tok]
        if tokens:
            return tokens[0].lower()
    if download_path:
        try:
            expanded = Path(download_path).expanduser()
            parts = list(expanded.parts)
            if "weights" in (part.lower() for part in parts):
                lowered_parts = [p.lower() for p in parts]
                idx = lowered_parts.index("weights")
                if idx + 1 < len(parts):
                    candidate = parts[idx + 1]
                    if candidate:
                        return candidate.lower()
            if parts:
                return parts[-2].lower() if len(parts) >= 2 else parts[0].lower()
        except Exception:  # noqa: BLE001
            pass
    return None


@dataclass
class RepositoryMetadata:
    """Normalized repository metadata for download routing."""

    owner: str
    repo_name: str
    branch: str
    repository_id: str
    storage_repo_name: str
    registry_model_name: str


@dataclass
class DownloadProgress:
    """Progress information for a download."""

    total_files: int
    completed_files: int
    total_bytes: int
    downloaded_bytes: int
    current_speed: str
    eta: str


logger = logging.getLogger(__name__)


class ModelDownloader:
    """Main model downloader class."""

    def __init__(self, config: Optional[DownloaderConfig] = None) -> None:
        self.config = config or get_config()
        # unified registry (JSON) only
        self._registry_cache = load_registry()
        self.url_analyzer = URLAnalyzer()
        self.format_detector = FormatDetector()

        # Progress tracking
        self._current_log_file: Optional[Path] = None
        self._current_file_count: int = 0

        # Ensure aria2c is available
        self._check_aria2c()

    def _log(self, message: str, *, level: int = logging.INFO) -> None:
        """Log a message and mirror it to stdout for CLI parity."""

        logger.log(level, message)
        print(message)

    def _check_aria2c(self) -> None:
        """Check if aria2c is available."""
        try:
            subprocess.run(
                ["aria2c", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "aria2c not found. Please install aria2c for optimal download performance.\n"
                "Ubuntu/Debian: sudo apt install aria2\n"
                "macOS: brew install aria2"
            )

    def _parse_aria2c_progress(
        self, log_file: Path, total_files: int
    ) -> Dict[str, Any]:
        """Parse aria2c progress by monitoring actual file sizes and aria2c activity."""

        # Initialize tracking variables if not present
        if not hasattr(self, "_progress_tracker"):
            self._progress_tracker = {
                "last_size_check": {},
                "last_check_time": time.time(),
                "last_speed": 0,
            }

        current_time = time.time()
        tracker = self._progress_tracker
        time_diff = current_time - tracker["last_check_time"]

        # Don't check too frequently for performance
        if time_diff < 0.5:
            return {
                "active_files": total_files,
                "total_files": total_files,
                "total_size_bytes": 0,
                "downloaded_bytes": 0,
                "speed_bps": tracker["last_speed"],
                "eta_seconds": 0,
            }

        try:
            total_size_change = 0
            completed_files = 0
            active_files = 0
            downloaded_bytes = 0

            # Monitor actual file sizes if we have target directory
            if hasattr(self, "_current_target_dir") and self._current_target_dir:
                for file_path in self._current_target_dir.rglob("*"):
                    if file_path.is_file():
                        file_key = str(file_path)
                        current_size = file_path.stat().st_size

                        # Skip aria2 control files for size calculation
                        if file_path.suffix == ".aria2":
                            active_files += 1
                            continue

                        # Calculate size change for speed estimation
                        last_size = tracker["last_size_check"].get(file_key, 0)
                        size_change = max(0, current_size - last_size)
                        total_size_change += size_change
                        tracker["last_size_check"][file_key] = current_size
                        downloaded_bytes += current_size

                        # Check if file is complete (no corresponding .aria2 file)
                        aria2_file = file_path.parent / f"{file_path.name}.aria2"
                        if not aria2_file.exists() and current_size > 0:
                            completed_files += 1
                        elif aria2_file.exists():
                            active_files += 1

            # Calculate speed in bytes per second
            speed_bps = int(total_size_change / time_diff) if time_diff > 0 else 0
            tracker["last_speed"] = speed_bps
            tracker["last_check_time"] = current_time

            return {
                "active_files": active_files,
                "total_files": total_files,
                "total_size_bytes": 0,
                "downloaded_bytes": downloaded_bytes,
                "speed_bps": speed_bps,
                "eta_seconds": 0,
            }

        except Exception:
            return {
                "active_files": total_files,
                "total_files": total_files,
                "total_size_bytes": 0,
                "downloaded_bytes": 0,
                "speed_bps": 0,
                "eta_seconds": 0,
            }

    def get_progress(self) -> Dict[str, Any]:
        """Get current download progress if a download is active."""
        if self._current_log_file and self._current_file_count > 0:
            return self._parse_aria2c_progress(
                self._current_log_file, self._current_file_count
            )
        return {
            "active_files": 0,
            "total_files": 0,
            "total_size_bytes": 0,
            "downloaded_bytes": 0,
            "speed_bps": 0,
            "eta_seconds": 0,
        }

    def _verify_download_complete(self, files: List[Any], target_dir: Path) -> bool:
        """Verify that all downloaded files are complete and valid."""
        for file_info in files:
            file_path = target_dir / file_info.path

            # Check if file exists
            if not file_path.exists():
                self._log(f"⚠️  Missing file: {file_info.path}", level=logging.WARNING)
                return False

            # Check for aria2 control file indicating incomplete download
            aria2_control = file_path.parent / f"{file_path.name}.aria2"
            if aria2_control.exists():
                self._log(
                    f"⚠️  Incomplete download detected: {file_info.path} (aria2 control file exists)",
                    level=logging.WARNING,
                )
                return False

            # Check file has non-zero size
            if file_path.stat().st_size == 0:
                self._log(f"⚠️  Zero-size file: {file_info.path}", level=logging.WARNING)
                return False

        return True

    def download(
        self,
        model_identifier: str,
        format_preference: Optional[Union[Sequence[str], str]] = None,
        location_override: Optional[str] = None,
        include_optional: bool = True,
        include_large_optional: bool = False,
        force_redownload: bool = False,
        interactive: bool = True,
        weight_variants: Optional[Sequence[str]] = None,
        support_repo: Optional[str] = None,
    ) -> RegistryEntry:
        """Download a model from a URL or HuggingFace identifier."""

        analysis = self.url_analyzer.analyze_url(model_identifier)
        repo_meta = self._build_repository_metadata(analysis)
        preferred_formats = self._normalise_format_preferences(format_preference)

        existing_entry = self._handle_existing_download(
            repo_meta.registry_model_name,
            force_redownload=force_redownload,
            interactive=interactive,
        )
        if existing_entry is not None:
            return existing_entry

        primary_format = self._detect_primary_format(analysis, preferred_formats)
        target_dir = self._resolve_target_dir(
            primary_format,
            location_override,
            repo_meta.owner,
            repo_meta.storage_repo_name,
        )

        weight_filter: Optional[Set[str]] = None
        if weight_variants:
            weight_filter = {v.strip() for v in weight_variants if v and v.strip()}
            if not weight_filter:
                weight_filter = None

        primary_categories = self._partition_files(
            analysis, weight_filter=weight_filter
        )
        if not primary_categories["weights"]:
            raise ValueError(
                "No model weight files matched the requested variants."
                if weight_filter
                else "No model weight files were detected in the repository."
            )

        support_categories: Optional[Dict[str, List[Any]]] = None
        support_repo_meta: Optional[RepositoryMetadata] = None
        if support_repo:
            self._log(f"🔗 Fetching support assets from {support_repo}")
            support_analysis = self.url_analyzer.analyze_url(support_repo)
            support_repo_meta = self._build_repository_metadata(support_analysis)
            support_categories = self._partition_files(
                support_analysis, weight_filter=set()
            )

        all_files = self._download_repository_assets(
            repo_meta=repo_meta,
            categories=primary_categories,
            target_dir=target_dir,
            include_optional=include_optional,
            include_large_optional=include_large_optional,
            include_weights=True,
        )

        selected_weight_paths: Sequence[str] = []
        if weight_filter:
            selected_weight_paths = [f.path for f in primary_categories["weights"]]

        if support_categories and support_repo_meta:
            support_files = self._download_repository_assets(
                repo_meta=support_repo_meta,
                categories=support_categories,
                target_dir=target_dir,
                include_optional=include_optional,
                include_large_optional=False,
                include_weights=False,
            )
            all_files.extend(support_files)

        self._ensure_verified(all_files, target_dir)
        chat_template_info = self._inspect_chat_templates(target_dir)
        total_size = self._calculate_total_size(all_files, target_dir)

        entry = self._register_download(
            repo_meta,
            primary_format,
            total_size,
            all_files,
            target_dir,
            analysis,
            chat_template_info,
            support_repo=support_repo if support_repo else None,
            selected_weights=selected_weight_paths,
        )

        return entry

    def _build_repository_metadata(self, analysis: Any) -> RepositoryMetadata:
        owner = analysis.repository.owner
        repo_name = analysis.repository.repo
        repository_id = f"{owner}/{repo_name}"
        branch = analysis.repository.branch or "main"
        storage_repo_name = (
            repo_name if branch == "main" else f"{repo_name}@{branch.replace('/', '_')}"
        )
        registry_model_name = (
            repository_id if branch == "main" else f"{repository_id}@{branch}"
        )

        return RepositoryMetadata(
            owner=owner,
            repo_name=repo_name,
            branch=branch,
            repository_id=repository_id,
            storage_repo_name=storage_repo_name,
            registry_model_name=registry_model_name,
        )

    def _normalise_format_preferences(
        self, format_preference: Optional[Union[Sequence[str], str]]
    ) -> Optional[List[str]]:
        if isinstance(format_preference, str):
            return [format_preference]
        if format_preference is None:
            return None
        return [str(fmt) for fmt in format_preference]

    def _handle_existing_download(
        self,
        registry_model_name: str,
        *,
        force_redownload: bool,
        interactive: bool,
    ) -> Optional[RegistryEntry]:
        if force_redownload:
            return None

        existing = self._registry_cache.get(registry_model_name)
        if not existing or not existing.download_path:
            return None

        existing_path = Path(existing.download_path).expanduser()
        if existing_path.exists():
            self._log(f"✅ Model already downloaded: {existing.name}")
            return existing

        self._log(
            f"⚠️  Model '{existing.name}' found in registry but directory missing: {existing.download_path}",
            level=logging.WARNING,
        )

        if not interactive:
            self._log("🔄 Non-interactive mode: proceeding with re-download...")
            return None

        try:
            response = input("Directory not found. Re-download the model? (y/N): ")
        except (KeyboardInterrupt, EOFError):
            self._log("\n❌ Download cancelled by user", level=logging.WARNING)
            return existing

        if response.strip().lower() not in {"y", "yes"}:
            self._log("❌ Download cancelled by user", level=logging.WARNING)
            return existing

        self._log("🔄 Proceeding with re-download...")
        return None

    def _detect_primary_format(
        self, analysis: Any, preferred_formats: Optional[List[str]]
    ) -> str:
        files: List[Any] = []
        for file_list in analysis.files.values():
            files.extend(file_list)

        formats = self.format_detector.detect_from_filelist([f.path for f in files])
        if not formats:
            self._log(
                "⚠️  Could not detect model format, using default routing",
                level=logging.WARNING,
            )
            return "unknown"

        if preferred_formats:
            for pref in preferred_formats:
                for fmt in formats:
                    if fmt.format_type == pref:
                        self._log(f"🔧 Detected format: {pref}")
                        return pref

        primary_format = formats[0].format_type
        self._log(f"🔧 Detected format: {primary_format}")
        return primary_format

    def _resolve_target_dir(
        self,
        primary_format: str,
        location_override: Optional[str],
        owner: str,
        storage_repo_name: str,
    ) -> Path:
        if location_override:
            normalized_override = location_override.strip()
            if normalized_override in {"linux_wsl", "windows_lmstudio"}:
                if normalized_override == "linux_wsl":
                    base_dir = self.config.linux_wsl.root / "weights"
                else:
                    base_dir = self.config.windows_lmstudio.root
                target_dir = base_dir / owner / storage_repo_name
            else:
                target_dir = (
                    Path(normalized_override).expanduser() / owner / storage_repo_name
                )
        else:
            target_dir = self.config.get_target_directory(
                primary_format,
                storage_repo_name,
                publisher=owner,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"📁 Target directory: {target_dir}")
        return target_dir

    def _partition_files(
        self, analysis: Any, weight_filter: Optional[Set[str]] = None
    ) -> Dict[str, List[Any]]:
        categories = {
            "weights": [],
            "config": [],
            "tokenizer": [],
            "optional": [],
            "large_optional": [],
        }

        filter_targets: Optional[Set[str]] = None
        if weight_filter:
            filter_targets = {item.lower() for item in weight_filter if item}

        def _matches_weight(file_info: Any) -> bool:
            if not filter_targets:
                return True
            path_lower = file_info.path.lower()
            if path_lower in filter_targets:
                return True
            name_lower = Path(file_info.path).name.lower()
            return name_lower in filter_targets

        matched_weight = False

        for category, file_list in analysis.files.items():
            if category == "model_weights":
                for file_info in file_list:
                    if _matches_weight(file_info):
                        categories["weights"].append(file_info)
                        matched_weight = True
                continue
            if category == "config":
                categories["config"].extend(file_list)
                continue
            if category == "tokenizer":
                categories["tokenizer"].extend(file_list)
                continue
            if category == "optional":
                categories["optional"].extend(file_list)
                continue
            if category == "large_optional":
                categories["large_optional"].extend(file_list)
                continue

        # If the caller supplied a filter but nothing matched, fall back to the full
        # weight set so we never end up with an incomplete checkpoint.
        if filter_targets and not matched_weight:
            categories["weights"] = list(analysis.files.get("model_weights", []))

        return categories

    def _download_repository_assets(
        self,
        *,
        repo_meta: RepositoryMetadata,
        categories: Dict[str, List[Any]],
        target_dir: Path,
        include_optional: bool,
        include_large_optional: bool,
        include_weights: bool,
    ) -> List[Any]:
        base_url = f"https://huggingface.co/{repo_meta.repository_id}/resolve/{repo_meta.branch}"

        weights = categories.get("weights", []) if include_weights else []
        config_files = categories.get("config", [])
        tokenizer_files = categories.get("tokenizer", [])
        optional_files = list(categories.get("optional", []))
        large_optional_files = list(categories.get("large_optional", []))

        minimal_docs: List[Any] = []
        seen_required = {f.path for f in weights + config_files + tokenizer_files}
        filtered_optional: List[Any] = []
        for file_info in optional_files:
            if _is_minimal_doc(file_info.path) and file_info.path not in seen_required:
                minimal_docs.append(file_info)
            else:
                filtered_optional.append(file_info)
        optional_files = filtered_optional

        required_files: List[Any] = []
        required_files.extend(weights)
        required_files.extend(config_files)
        required_files.extend(tokenizer_files)

        def _download_if_needed(file_list: List[Any], description: str) -> None:
            if not file_list:
                return
            files_to_fetch = []
            for info in file_list:
                target_path = target_dir / info.path
                if target_path.exists():
                    continue
                files_to_fetch.append(info)
            if not files_to_fetch:
                return
            self._log(f"📥 Downloading {len(files_to_fetch)} {description}...")
            self._download_files_with_aria2c(files_to_fetch, base_url, target_dir)

        _download_if_needed(required_files, "required files")
        _download_if_needed(minimal_docs, "repository docs (README/LICENSE)")

        if include_optional:
            _download_if_needed(optional_files, "optional files")

        if include_large_optional:
            _download_if_needed(large_optional_files, "large optional files")

        combined: List[Any] = []
        combined.extend(required_files)
        combined.extend(minimal_docs)
        if include_optional:
            combined.extend(optional_files)
        if include_large_optional:
            combined.extend(large_optional_files)
        return combined

    def _ensure_verified(self, files: List[Any], target_dir: Path) -> None:
        if not self._verify_download_complete(files, target_dir):
            raise RuntimeError(
                "Download verification failed - some files are missing, incomplete, or corrupted. "
                "Please try the download again."
            )
        self._log("✅ Download verification passed - all files complete")

    def _inspect_chat_templates(self, target_dir: Path) -> Dict[str, Any]:
        tokenizer_cfg_path = target_dir / "tokenizer_config.json"
        has_embedded_template = False
        embedded_template_snippet = None

        if tokenizer_cfg_path.exists():
            try:
                import json

                data = json.loads(tokenizer_cfg_path.read_text(encoding="utf-8"))
                tmpl = data.get("chat_template")
                if tmpl and isinstance(tmpl, str) and "{{" in tmpl:
                    has_embedded_template = True
                    embedded_template_snippet = tmpl[:160] + (
                        "..." if len(tmpl) > 160 else ""
                    )
            except Exception:  # noqa: BLE001
                pass
        else:
            self._log(
                "ℹ️  tokenizer_config.json not present; cannot inspect chat template",
                level=logging.INFO,
            )

        template_files: List[str] = []
        candidate_patterns = [
            ".jinja",
            "chat_template.json",
            "chat_template",
            "template",
        ]

        try:
            for path in target_dir.iterdir():
                if not path.is_file():
                    continue
                name_lower = path.name.lower()
                if any(pattern in name_lower for pattern in candidate_patterns):
                    if path.stat().st_size < 2_000_000:
                        try:
                            head = path.read_text(encoding="utf-8", errors="ignore")[
                                :400
                            ]
                            if "{{" in head and "}}" in head:
                                template_files.append(path.name)
                        except Exception:  # noqa: BLE001
                            pass
        except Exception:  # noqa: BLE001
            pass

        has_external_template = len(template_files) > 0
        has_chat_template = has_embedded_template or has_external_template

        if has_chat_template:
            detail = "embedded" if has_embedded_template else "external-file"
            self._log(
                f"💬 Chat template detected ({detail}). External files: {template_files if template_files else 'n/a'}"
            )
        else:
            self._log(
                "💬 No chat template detected (embedded or external). If you encounter 400 errors ('default chat template no longer allowed'), use --chat-template to supply one.",
                level=logging.INFO,
            )

        return {
            "has_chat_template": has_chat_template,
            "has_embedded_chat_template": has_embedded_template,
            "embedded_chat_template_preview": embedded_template_snippet,
            "external_chat_template_files": template_files,
        }

    def _calculate_total_size(self, files: List[Any], target_dir: Path) -> int:
        return sum((target_dir / f.path).stat().st_size for f in files)

    def _determine_location_label(self, target_dir: Path) -> str:
        if (
            self.config.linux_wsl.root in target_dir.parents
            or self.config.linux_wsl.root == target_dir
        ):
            return "linux_wsl"
        if (
            self.config.windows_lmstudio.root in target_dir.parents
            or self.config.windows_lmstudio.root == target_dir
        ):
            return "windows_lmstudio"
        return "custom"

    def _register_download(
        self,
        repo_meta: RepositoryMetadata,
        primary_format: str,
        total_size: int,
        files: List[Any],
        target_dir: Path,
        analysis: Any,
        chat_template_info: Dict[str, Any],
        support_repo: Optional[str],
        selected_weights: Sequence[str],
    ) -> RegistryEntry:
        location_label = self._determine_location_label(target_dir)
        downloaded_files = [f.path for f in files]

        det_fmt, det_quant = detect_format_and_quant(target_dir)
        final_format = det_fmt or primary_format

        readme_path = _select_primary_readme(target_dir)
        readme_rel: Optional[str] = None
        readme_sha: Optional[str] = None
        readme_signals: Optional[Dict[str, Any]] = None
        if readme_path:
            try:
                readme_rel = readme_path.relative_to(target_dir).as_posix()
            except ValueError:
                readme_rel = readme_path.name
            readme_sha = _hash_small_file(readme_path)
            read_text = _read_text_limited(readme_path)
            readme_signals = _parse_readme_signals(read_text) if read_text else {}

        license_candidates = [
            p
            for p in target_dir.rglob("*")
            if p.is_file() and p.name.lower().startswith("license")
        ]
        license_rel: Optional[str] = None
        if license_candidates:
            try:
                license_rel = (
                    (
                        sorted(
                            license_candidates,
                            key=lambda p: (
                                len(p.relative_to(target_dir).parts),
                                p.as_posix(),
                            ),
                        )[0]
                    )
                    .relative_to(target_dir)
                    .as_posix()
                )
            except Exception:  # noqa: BLE001
                license_rel = license_candidates[0].name

        # Auto-assign backend based on container. vLLM handles safetensors broadly,
        # including AWQ, GPTQ, FP8, and SqueezeLLM variants.
        auto_backend = None
        fmt_lower = (final_format or "").lower()
        if fmt_lower == "safetensors":
            auto_backend = "vllm"

        try:
            entry = record_download(
                hf_id=repo_meta.repository_id,
                backend=auto_backend or "unassigned",
                format_type=final_format,
                quantization=det_quant,
                path=str(target_dir),
                location=location_label,
                files=downloaded_files,
                size_bytes=total_size,
                source_provider="hf",
                roles=[],
                role_priority=None,
            )

            entry.metadata = entry.metadata or {}
            if readme_rel:
                entry.metadata["readme_file"] = readme_rel
                entry.metadata["readme_sha256"] = readme_sha
                entry.metadata["readme_signals"] = readme_signals or {}
            else:
                entry.metadata.pop("readme_file", None)
                entry.metadata.pop("readme_sha256", None)
                entry.metadata.pop("readme_signals", None)
            if license_rel:
                entry.metadata["license_file"] = license_rel
            else:
                entry.metadata.pop("license_file", None)

            producer_value = _derive_producer(
                existing=entry.metadata.get("producer") if entry.metadata else None,
                hf_id=repo_meta.repository_id,
                source_provider=entry.source_provider,
                served_model_id=entry.served_model_id,
                download_path=entry.download_path,
            )
            if producer_value and not entry.metadata.get("producer"):
                entry.metadata["producer"] = producer_value

            if support_repo:
                entry.metadata["support_repository"] = support_repo
            else:
                entry.metadata.pop("support_repository", None)

            if selected_weights:
                entry.metadata["selected_weight_files"] = list(selected_weights)
            else:
                entry.metadata.pop("selected_weight_files", None)

            entry.metadata.update(
                {
                    "model_type": analysis.repository.model_type,
                    "library": analysis.repository.library_name,
                    "files_downloaded": len(files),
                    "verified_complete": True,
                    "has_chat_template": chat_template_info["has_chat_template"],
                    "has_embedded_chat_template": chat_template_info[
                        "has_embedded_chat_template"
                    ],
                    "external_chat_template_files": chat_template_info[
                        "external_chat_template_files"
                    ],
                    "embedded_chat_template_preview": chat_template_info[
                        "embedded_chat_template_preview"
                    ],
                    "branch": repo_meta.branch,
                }
            )

            # Promote external chat template to primary chat_template if no embedded template
            try:
                if (not entry.chat_template.path) and chat_template_info[
                    "external_chat_template_files"
                ]:
                    first_path = chat_template_info["external_chat_template_files"][0]
                    tpl_file = target_dir / first_path
                    if tpl_file.exists() and tpl_file.is_file():
                        data = tpl_file.read_text(encoding="utf-8", errors="ignore")
                        sha = hashlib.sha256(data.encode("utf-8")).hexdigest()
                        # Update chat_template inline (source=external)
                        entry.chat_template.source = "external"
                        entry.chat_template.path = str(tpl_file)
                        entry.chat_template.sha256 = sha
                        entry.metadata["primary_chat_template_file"] = first_path
                        entry.metadata["primary_chat_template_sha256"] = sha
            except Exception:  # noqa: BLE001
                # Non-fatal; template promotion best-effort
                pass

            update_entries([entry], save=True)
            self._registry_cache = load_registry(force=True)  # type: ignore[arg-type]
            self._log(f"✅ Registry updated for {entry.name}")
            return entry
        except Exception as exc:  # noqa: BLE001
            self._log(
                f"⚠️  Warning: Could not update unified registry via adapter: {exc}",
                level=logging.WARNING,
            )
            raise

    def _download_files_with_aria2c(
        self, files: List[Any], base_url: str, target_dir: Path
    ) -> None:
        """Download files using aria2c."""

        if not files:
            return

        # Store target directory for progress monitoring
        self._current_target_dir = target_dir

        # Create temporary input file for aria2c
        with tempfile.NamedTemporaryFile(mode="w", suffix=".aria2", delete=False) as f:
            aria_input_file = f.name

            for file_info in files:
                file_url = f"{base_url}/{file_info.path}"
                local_path = target_dir / file_info.path

                # Ensure local directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Write aria2c input line with relative path (since we use cwd)
                f.write(f"{file_url}\n")
                f.write(f"  out={file_info.path}\n")

        try:
            # Build aria2c command
            log_file = target_dir / ".aria2c.log"
            aria_cmd = [
                "aria2c",
                "--input-file",
                aria_input_file,
                "--max-connection-per-server",
                str(self.config.max_connections_per_server),
                "--max-concurrent-downloads",
                str(self.config.max_concurrent_downloads),
                f"--continue={'true' if self.config.enable_resume else 'false'}",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--summary-interval",
                "0",
                "--truncate-console-readout=true",
                "--human-readable=true",
                "--console-log-level",
                "warn",
                # Clean, minimal progress display - warn level hides NOTICE messages
            ]

            # Execute aria2c with progress monitoring
            self._log(f"🚀 Starting parallel download of {len(files)} files...")

            # Store progress callback for CLI integration
            self._current_log_file = log_file
            self._current_file_count = len(files)

            # Execute aria2c from target directory to avoid path issues
            result = subprocess.run(
                aria_cmd, capture_output=False, check=True, cwd=str(target_dir)
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, aria_cmd)

            # Clean up log file after successful download
            if log_file.exists():
                log_file.unlink()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"aria2c download failed: {e}")
        finally:
            # Clean up aria2c input file
            try:
                os.unlink(aria_input_file)
            except OSError:
                pass

            # Reset progress tracking
            self._current_log_file = None
            self._current_file_count = 0

    def verify_model(self, model_name: str) -> bool:
        entry = self._registry_cache.get(model_name)
        if not entry or not entry.download_path:
            return False
        path = Path(entry.download_path).expanduser()
        if not path.exists():
            return False
        # optional: verify directory checksum if present
        if entry.download_directory_checksum:
            if _calculate_directory_checksum(path) != entry.download_directory_checksum:
                return False
        return True

    def remove_model(
        self,
        model_name: str,
        format_type: Optional[str] = None,
        location: Optional[str] = None,
        delete_files: bool = False,
    ) -> bool:
        """Remove model from registry and optionally delete files."""
        entry = self._registry_cache.get(model_name)
        if not entry:
            return False
        path = Path(entry.download_path) if entry.download_path else None
        if delete_files and path and path.exists():
            try:
                shutil.rmtree(path)
                print(f"🗑️  Deleted files: {path}")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Could not delete files: {e}")
                return False
        # If entry is only a download placeholder (unassigned backend & no roles) remove entirely
        if entry.backend == "unassigned" and not entry.roles:
            removed = remove_entry(model_name, save=True)
            if removed:
                self._registry_cache = load_registry(force=True)  # type: ignore[arg-type]
            return removed
        # Otherwise just clear download fields
        entry.download_format = None
        entry.download_location = None
        entry.download_path = None
        entry.download_size_bytes = None
        entry.download_files = []
        entry.download_directory_checksum = None
        update_entries([entry], save=True)
        self._registry_cache = load_registry(force=True)  # type: ignore[arg-type]
        return True

    def list_models(
        self,
        format_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> List[RegistryEntry]:
        """List downloaded models with optional filtering."""
        models = [
            e for e in self._registry_cache.values() if e.download_path is not None
        ]
        if format_filter:
            models = [m for m in models if m.download_format == format_filter]
        if location_filter:
            models = [m for m in models if m.download_location == location_filter]
        return models

    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        models = [
            e for e in self._registry_cache.values() if e.download_path is not None
        ]
        total_size = sum((m.download_size_bytes or 0) for m in models)
        format_counts = {}
        location_counts = {}

        for model in models:
            fmt = model.download_format or "unknown"
            loc = model.download_location or "unknown"
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
            location_counts[loc] = location_counts.get(loc, 0) + 1

        return {
            "total_models": len(models),
            "total_size_bytes": total_size,
            "by_format": format_counts,
            "by_location": location_counts,
        }


def _calculate_directory_checksum(directory: Path) -> str:
    """Lightweight directory checksum (names + sizes) first 16 chars of sha256."""
    import hashlib

    if not directory.exists():
        return ""
    hasher = hashlib.sha256()
    files = sorted(directory.rglob("*"))
    for file_path in files:
        if file_path.is_file():
            rel_path = file_path.relative_to(directory)
            hasher.update(str(rel_path).encode())
            hasher.update(str(file_path.stat().st_size).encode())
    return hasher.hexdigest()[:16]

"""
Deployment Profile Manager

Manages deployment profiles based on GPU hardware detection and configuration.
Provides profile selection logic with auto-detection and manual override support.
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from imageworks.libs.hardware.gpu_detector import GPUDetector, GPUInfo

logger = logging.getLogger(__name__)


@dataclass
class DeploymentProfile:
    """Represents a deployment profile configuration."""

    name: str
    max_vram_mb: int
    max_concurrent_models: int
    preferred_quantizations: List[str]
    strategy: str
    autostart_timeout_seconds: int
    grace_period_seconds: int
    model_selection_bias: str
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, config: Dict) -> "DeploymentProfile":
        """Create a DeploymentProfile from a config dictionary."""
        return cls(
            name=name,
            max_vram_mb=config["max_vram_mb"],
            max_concurrent_models=config["max_concurrent_models"],
            preferred_quantizations=config["preferred_quantizations"],
            strategy=config["strategy"],
            autostart_timeout_seconds=config["autostart_timeout_seconds"],
            grace_period_seconds=config["grace_period_seconds"],
            model_selection_bias=config["model_selection_bias"],
            description=config.get("description", ""),
        )

    def to_dict(self) -> Dict:
        """Convert profile to dictionary representation."""
        return {
            "name": self.name,
            "max_vram_mb": self.max_vram_mb,
            "max_concurrent_models": self.max_concurrent_models,
            "preferred_quantizations": self.preferred_quantizations,
            "strategy": self.strategy,
            "autostart_timeout_seconds": self.autostart_timeout_seconds,
            "grace_period_seconds": self.grace_period_seconds,
            "model_selection_bias": self.model_selection_bias,
            "description": self.description,
        }


class ProfileManager:
    """
    Manages deployment profiles with GPU auto-detection and manual override.

    Features:
    - Loads profiles from pyproject.toml
    - Auto-detects GPU hardware and recommends appropriate profile
    - Supports manual override via IMAGEWORKS_DEPLOYMENT_PROFILE env var
    - Caches active profile for performance
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ProfileManager.

        Args:
            config_path: Path to pyproject.toml. Defaults to workspace root.
        """
        if config_path is None:
            # Find pyproject.toml in workspace root
            workspace_root = Path(__file__).parent.parent.parent.parent
            config_path = workspace_root / "pyproject.toml"

        self.config_path = config_path
        self.gpu_detector = GPUDetector()
        self._profiles: Dict[str, DeploymentProfile] = {}
        self._active_profile: Optional[DeploymentProfile] = None
        self._detected_gpus: Optional[List[GPUInfo]] = None

        self._load_profiles()
        self._initialize_active_profile()

    def _load_profiles(self) -> None:
        """Load deployment profiles from pyproject.toml."""
        try:
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)

            profiles_config = (
                config.get("tool", {})
                .get("imageworks", {})
                .get("deployment_profiles", {})
            )

            if not profiles_config:
                logger.warning("No deployment profiles found in pyproject.toml")
                return

            for profile_name, profile_data in profiles_config.items():
                try:
                    profile = DeploymentProfile.from_dict(profile_name, profile_data)
                    self._profiles[profile_name] = profile
                    logger.debug(f"Loaded profile: {profile_name}")
                except Exception as e:
                    logger.error(f"Failed to load profile {profile_name}: {e}")

            logger.info(f"Loaded {len(self._profiles)} deployment profiles")

        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")

    def _initialize_active_profile(self) -> None:
        """Initialize active profile based on env var or GPU auto-detection."""
        # Check for manual override
        manual_profile = os.getenv("IMAGEWORKS_DEPLOYMENT_PROFILE")

        if manual_profile:
            if manual_profile in self._profiles:
                self._active_profile = self._profiles[manual_profile]
                logger.info(f"Using manually specified profile: {manual_profile}")
                return
            else:
                logger.warning(
                    f"Manual profile '{manual_profile}' not found. "
                    f"Available: {list(self._profiles.keys())}. Falling back to auto-detection."
                )

        # Auto-detect GPU and recommend profile
        try:
            self._detected_gpus = self.gpu_detector.detect_gpus()
            recommended_profile_name = self.gpu_detector.recommend_profile()

            if recommended_profile_name and recommended_profile_name in self._profiles:
                self._active_profile = self._profiles[recommended_profile_name]
                logger.info(
                    f"Auto-detected profile: {recommended_profile_name} "
                    f"(GPUs: {len(self._detected_gpus)}, "
                    f"VRAM: {self.gpu_detector.get_usable_vram_mb()}MB)"
                )
            else:
                logger.warning(
                    f"Recommended profile '{recommended_profile_name}' not found. "
                    f"Using fallback."
                )
                self._use_fallback_profile()

        except Exception as e:
            logger.error(f"GPU detection failed: {e}. Using fallback profile.")
            self._use_fallback_profile()

    def _use_fallback_profile(self) -> None:
        """Use a fallback profile when auto-detection fails."""
        # Try to use 'development' profile as fallback, or first available
        fallback_name = (
            "development"
            if "development" in self._profiles
            else next(iter(self._profiles), None)
        )

        if fallback_name:
            self._active_profile = self._profiles[fallback_name]
            logger.info(f"Using fallback profile: {fallback_name}")
        else:
            logger.error(
                "No profiles available! System cannot operate without a profile."
            )

    def get_active_profile(self) -> Optional[DeploymentProfile]:
        """Get the currently active deployment profile."""
        return self._active_profile

    def get_profile(self, name: str) -> Optional[DeploymentProfile]:
        """Get a specific profile by name."""
        return self._profiles.get(name)

    def get_all_profiles(self) -> Dict[str, DeploymentProfile]:
        """Get all available profiles."""
        return self._profiles.copy()

    def get_detected_gpus(self) -> Optional[List[GPUInfo]]:
        """Get detected GPU information."""
        if self._detected_gpus is None:
            # Lazy detection if not already done
            try:
                self._detected_gpus = self.gpu_detector.detect_gpus()
            except Exception as e:
                logger.error(f"GPU detection failed: {e}")

        return self._detected_gpus

    def set_active_profile(self, profile_name: str) -> bool:
        """
        Manually set the active profile.

        Args:
            profile_name: Name of the profile to activate

        Returns:
            True if profile was set successfully, False otherwise
        """
        if profile_name not in self._profiles:
            logger.error(f"Profile '{profile_name}' not found")
            return False

        self._active_profile = self._profiles[profile_name]
        logger.info(f"Active profile changed to: {profile_name}")
        return True

    def get_profile_info(self) -> Dict:
        """
        Get comprehensive info about active profile and detected hardware.

        Returns:
            Dictionary with profile and GPU information
        """
        info = {
            "active_profile": None,
            "detected_gpus": [],
            "available_profiles": list(self._profiles.keys()),
            "manual_override": os.getenv("IMAGEWORKS_DEPLOYMENT_PROFILE"),
        }

        if self._active_profile:
            info["active_profile"] = self._active_profile.to_dict()

        gpus = self.get_detected_gpus()
        if gpus:
            info["detected_gpus"] = [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_vram_mb": gpu.vram_total_mb,
                    "free_vram_mb": gpu.vram_free_mb,
                    "compute_capability": gpu.compute_capability,
                    "uuid": gpu.uuid,
                }
                for gpu in gpus
            ]
            info["total_usable_vram_mb"] = self.gpu_detector.get_usable_vram_mb()

        return info

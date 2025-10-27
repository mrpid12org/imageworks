"""
Role-Based Model Selector

Provides intelligent model selection based on task roles and deployment profiles.
Filters models by VRAM constraints and sorts by role suitability.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from imageworks.chat_proxy.profile_manager import DeploymentProfile

logger = logging.getLogger(__name__)


class RoleSelector:
    """
    Selects optimal models based on task roles and deployment constraints.

    Features:
    - Loads model registry with role priority metadata
    - Filters models by VRAM constraints from active profile
    - Sorts by role_priority scores and quantization preferences
    - Supports multiple model roles: keywords, caption, description, narration, etc.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize RoleSelector.

        Args:
            registry_path: Path to model_registry.curated.json. Defaults to configs/.
        """
        if registry_path is None:
            workspace_root = Path(__file__).parent.parent.parent.parent
            registry_path = workspace_root / "configs" / "model_registry.curated.json"

        self.registry_path = registry_path
        self._models: List[Dict] = []

        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from JSON file."""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            # Handle both list format and object format
            if isinstance(data, list):
                self._models = data
            else:
                self._models = data.get("models", [])

            logger.info(f"Loaded {len(self._models)} models from registry")

        except FileNotFoundError:
            logger.error(f"Registry file not found: {self.registry_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registry JSON: {e}")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def select_for_role(
        self, role: str, profile: DeploymentProfile, top_n: int = 3
    ) -> List[Dict]:
        """
        Select best models for a specific role within profile constraints.

        Args:
            role: The task role (e.g., "keywords", "caption", "description")
            profile: Active deployment profile with VRAM constraints
            top_n: Number of top models to return

        Returns:
            List of model dictionaries sorted by suitability (best first)
        """
        # Filter models by role and VRAM constraints
        candidates = []

        for model in self._models:
            # Check if model has role priority
            role_priority = model.get("role_priority", {}).get(role)
            if role_priority is None:
                continue

            # Check VRAM constraint
            vram_estimate = model.get("vram_estimate_mb", float("inf"))
            if vram_estimate > profile.max_vram_mb:
                continue

            candidates.append(model)

        # Sort by role priority (higher is better) and quantization preference
        def sort_key(model: Dict) -> Tuple[int, int, int]:
            # Primary: role_priority score (higher first)
            role_score = model.get("role_priority", {}).get(role, 0)

            # Secondary: quantization preference (earlier in list = higher priority)
            quantization = model.get("quantization", "").lower()
            quant_priority = 0
            for i, pref_quant in enumerate(profile.preferred_quantizations):
                if pref_quant.lower() in quantization:
                    quant_priority = len(profile.preferred_quantizations) - i
                    break

            # Tertiary: lower VRAM is better (leave room for other models)
            vram_inverse = -model.get("vram_estimate_mb", 0)

            return (role_score, quant_priority, vram_inverse)

        candidates.sort(key=sort_key, reverse=True)

        # Apply model_selection_bias from profile
        if profile.model_selection_bias in ("vram_efficient", "efficiency"):
            # Re-sort to prefer lower VRAM when scores are similar
            candidates.sort(
                key=lambda m: (
                    m.get("role_priority", {}).get(role, 0),
                    -m.get("vram_estimate_mb", float("inf")),
                ),
                reverse=True,
            )
        elif profile.model_selection_bias == "quality":
            # Already sorted by role_priority first, which is the quality metric
            pass

        logger.debug(
            f"Role '{role}' selection: {len(candidates)} candidates "
            f"(max VRAM: {profile.max_vram_mb}MB, bias: {profile.model_selection_bias})"
        )

        return candidates[:top_n]

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by its ID."""
        for model in self._models:
            if model.get("id") == model_id:
                return model
        return None

    def get_available_roles(self) -> List[str]:
        """
        Get list of all available roles across all models.

        Returns:
            Sorted list of unique role names
        """
        roles = set()
        for model in self._models:
            role_priority = model.get("role_priority", {})
            roles.update(role_priority.keys())

        return sorted(roles)

    def get_models_for_profile(self, profile: DeploymentProfile) -> List[Dict]:
        """
        Get all models that fit within profile VRAM constraints.

        Args:
            profile: Deployment profile with constraints

        Returns:
            List of models that fit within the profile's VRAM limit
        """
        fitting_models = [
            model
            for model in self._models
            if model.get("vram_estimate_mb", float("inf")) <= profile.max_vram_mb
        ]

        logger.debug(
            f"Profile '{profile.name}': {len(fitting_models)}/{len(self._models)} "
            f"models fit within {profile.max_vram_mb}MB VRAM limit"
        )

        return fitting_models

    def explain_selection(
        self, role: str, profile: DeploymentProfile, model_id: str
    ) -> Dict:
        """
        Explain why a model was (or wasn't) selected for a role.

        Args:
            role: Task role
            profile: Deployment profile
            model_id: Model ID to explain

        Returns:
            Dictionary with selection reasoning
        """
        model = self.get_model_by_id(model_id)

        if not model:
            return {
                "model_id": model_id,
                "status": "not_found",
                "reason": "Model not found in registry",
            }

        explanation = {
            "model_id": model_id,
            "model_name": model.get("name", "Unknown"),
            "role": role,
            "profile": profile.name,
            "checks": [],
        }

        # Check 1: Has role priority
        role_priority = model.get("role_priority", {}).get(role)
        if role_priority is None:
            explanation["status"] = "rejected"
            explanation["reason"] = f"Model does not support role '{role}'"
            explanation["checks"].append(
                {
                    "check": "role_support",
                    "passed": False,
                    "details": f"No role_priority defined for '{role}'",
                }
            )
            return explanation

        explanation["checks"].append(
            {
                "check": "role_support",
                "passed": True,
                "details": f"Role priority: {role_priority}",
            }
        )

        # Check 2: VRAM constraint
        vram_estimate = model.get("vram_estimate_mb", float("inf"))
        vram_ok = vram_estimate <= profile.max_vram_mb

        explanation["checks"].append(
            {
                "check": "vram_constraint",
                "passed": vram_ok,
                "details": f"Model VRAM: {vram_estimate}MB, Profile limit: {profile.max_vram_mb}MB",
            }
        )

        if not vram_ok:
            explanation["status"] = "rejected"
            explanation["reason"] = "Exceeds VRAM limit"
            return explanation

        # Check 3: Quantization preference
        quantization = model.get("quantization", "unknown")
        quant_match = any(
            pref.lower() in quantization.lower()
            for pref in profile.preferred_quantizations
        )

        explanation["checks"].append(
            {
                "check": "quantization_preference",
                "passed": quant_match,
                "details": (
                    f"Model quantization: {quantization}, "
                    f"Preferred: {', '.join(profile.preferred_quantizations)}"
                ),
            }
        )

        explanation["status"] = "eligible"
        explanation["reason"] = "Meets all constraints"

        return explanation

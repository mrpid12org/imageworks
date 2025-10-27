"""
GPU Detection Module.

Detects NVIDIA GPUs via nvidia-smi and recommends appropriate deployment profiles
based on available VRAM and GPU count.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Single GPU information."""

    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: tuple[int, int]
    uuid: str


class GPUDetector:
    """Detect and analyze available GPU hardware."""

    def __init__(self):
        self._cached_info: Optional[List[GPUInfo]] = None

    def detect_gpus(self) -> List[GPUInfo]:
        """
        Query nvidia-smi for GPU information.

        Returns:
            List of GPUInfo objects, empty list if no GPUs or nvidia-smi unavailable.
        """
        if self._cached_info is not None:
            return self._cached_info

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,compute_cap,uuid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    logger.warning(f"Skipping malformed nvidia-smi line: {line}")
                    continue

                index, name, vram_total, vram_free, compute_cap, uuid = parts

                try:
                    major, minor = map(int, compute_cap.split("."))
                except ValueError:
                    logger.warning(f"Invalid compute capability: {compute_cap}")
                    major, minor = 0, 0

                gpus.append(
                    GPUInfo(
                        index=int(index),
                        name=name,
                        vram_total_mb=int(float(vram_total)),
                        vram_free_mb=int(float(vram_free)),
                        compute_capability=(major, minor),
                        uuid=uuid,
                    )
                )

            self._cached_info = gpus
            logger.info(f"Detected {len(gpus)} GPU(s)")
            for gpu in gpus:
                logger.info(
                    f"  GPU {gpu.index}: {gpu.name} ({gpu.vram_total_mb}MB VRAM)"
                )

            return gpus

        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi query timed out")
            return []
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.info(f"No NVIDIA GPU detected or nvidia-smi unavailable: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error detecting GPUs: {e}")
            return []

    def get_usable_vram_mb(self, headroom_per_gpu_mb: int = 2048) -> int:
        """
        Calculate total usable VRAM across all GPUs.

        Args:
            headroom_per_gpu_mb: VRAM headroom to reserve per GPU (default 2GB).

        Returns:
            Total usable VRAM in MB.
        """
        gpus = self.detect_gpus()
        if not gpus:
            return 0

        total = sum(gpu.vram_total_mb for gpu in gpus)
        headroom = headroom_per_gpu_mb * len(gpus)
        return max(0, total - headroom)

    def recommend_profile(self) -> str:
        """
        Auto-recommend deployment profile based on detected hardware.

        Returns:
            Profile name (e.g., "constrained_16gb", "development").
        """
        gpus = self.detect_gpus()

        if not gpus:
            logger.info("No GPU detected, recommending 'development' profile")
            return "development"

        gpu_count = len(gpus)
        usable_vram = self.get_usable_vram_mb()

        logger.info(
            f"Hardware: {gpu_count} GPU(s), {usable_vram}MB usable VRAM (after headroom)"
        )

        # Single GPU profiles
        if gpu_count == 1:
            if usable_vram >= 85000:
                profile = "generous_96gb"
            elif usable_vram >= 20000:
                profile = "balanced_24gb"
            elif usable_vram >= 14000:
                profile = "constrained_16gb"
            else:
                profile = "development"

            logger.info(f"Recommended profile: {profile}")
            return profile

        # Multi-GPU profiles
        elif gpu_count == 2:
            # Check if GPUs are similar (within 20% VRAM)
            vram_list = [gpu.vram_total_mb for gpu in gpus]
            vram_variance = (
                max(vram_list) / min(vram_list) if min(vram_list) > 0 else 999
            )

            if vram_variance < 1.2:  # Similar GPUs
                avg_vram = sum(vram_list) / gpu_count
                if avg_vram >= 14000:
                    profile = "multi_gpu_2x16gb"
                else:
                    profile = "development"
            else:
                # Mismatched GPUs, use primary GPU rules
                primary_vram = gpus[0].vram_total_mb - 2048
                if primary_vram >= 20000:
                    profile = "balanced_24gb"
                elif primary_vram >= 14000:
                    profile = "constrained_16gb"
                else:
                    profile = "development"

            logger.info(f"Recommended profile: {profile}")
            return profile

        else:
            # 3+ GPUs: use total VRAM approach
            if usable_vram >= 60000:
                profile = "generous_96gb"
            elif usable_vram >= 40000:
                profile = "balanced_24gb"
            else:
                profile = "constrained_16gb"

            logger.info(f"Recommended profile: {profile}")
            return profile

    def clear_cache(self):
        """Clear cached GPU information (force re-detection)."""
        self._cached_info = None

"""Similarity strategies combining deterministic and embedding approaches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from .config import SimilarityConfig
from .embeddings import EmbeddingModel, EmbeddingError, create_embedding_model
from .models import StrategyMatch

logger = logging.getLogger(__name__)


class SimilarityStrategy:
    """Base class for similarity scoring strategies."""

    name: str

    def prime(self, library_paths: Sequence[Path]) -> None:
        """Prepare any state derived from the historical library."""

    def find_matches(self, candidate: Path, *, top_k: int) -> List[StrategyMatch]:
        """Return top *top_k* matches for the candidate image."""
        raise NotImplementedError


@dataclass
class EmbeddingSimilarityStrategy(SimilarityStrategy):
    """Similarity scoring based on image embeddings."""

    config: SimilarityConfig
    cache_path: Path
    metric: str = "cosine"

    def __post_init__(self) -> None:
        self.name = "embedding"
        self._embedding_model: EmbeddingModel | None = None
        self._library_vectors: Dict[Path, np.ndarray] = {}
        self.metric = self.metric.lower()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Library preparation
    # ------------------------------------------------------------------
    def prime(self, library_paths: Sequence[Path]) -> None:  # noqa: D401 - see base
        embedding_model = self._get_model()
        cache = self._load_cache()

        to_encode: List[Path] = []
        for path in library_paths:
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                logger.debug("Skipping missing library image: %s", path)
                continue

            cached = cache.get(key)
            if cached and cached.get("mtime") == mtime:
                vector = np.asarray(cached["vector"], dtype=np.float32)
                self._library_vectors[path] = EmbeddingModel._normalise(vector)
            else:
                to_encode.append(path)

        if to_encode:
            logger.info(
                "embedding_index_build",
                extra={
                    "event_type": "embedding_index_build",
                    "count": len(to_encode),
                    "strategy": self.name,
                },
            )
            try:
                encoded = embedding_model.batch_embed(to_encode)
            except Exception as exc:  # noqa: BLE001 - resilient runtime
                logger.error("Embedding generation failed: %s", exc)
                encoded = {}
            for path, vector in encoded.items():
                self._library_vectors[path] = EmbeddingModel._normalise(vector)
                try:
                    mtime = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                cache[str(path)] = {
                    "vector": vector.astype(np.float32).tolist(),
                    "mtime": mtime,
                }
            self._save_cache(cache)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def find_matches(self, candidate: Path, *, top_k: int) -> List[StrategyMatch]:
        embedding_model = self._get_model()
        try:
            vector = EmbeddingModel._normalise(embedding_model.embed(candidate))
        except Exception as exc:  # noqa: BLE001 - ensure graceful degradation
            logger.error("Failed to embed candidate %s: %s", candidate, exc)
            return []

        matches: List[StrategyMatch] = []
        for reference, ref_vector in self._library_vectors.items():
            score = self._compute_similarity(vector, ref_vector)
            matches.append(
                StrategyMatch(
                    candidate=candidate,
                    reference=reference,
                    score=score,
                    strategy=self.name,
                    reason=f"cosine similarity {score:.3f}" if self.metric == "cosine" else "embedding comparison",
                    extra={"metric": self.metric},
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            try:
                self._embedding_model = create_embedding_model(self.config)
            except EmbeddingError as exc:
                raise EmbeddingError(
                    f"Failed to initialise embedding backend '{self.config.embedding_backend}': {exc}"
                ) from exc
        return self._embedding_model

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            return float(np.dot(a, b))
        if self.metric == "dot":
            return float(np.dot(a, b))
        if self.metric == "euclidean":
            distance = float(np.linalg.norm(a - b))
            return float(1.0 / (1.0 + distance))
        if self.metric == "manhattan":
            distance = float(np.abs(a - b).sum())
            return float(1.0 / (1.0 + distance))
        raise ValueError(f"Unsupported similarity metric '{self.metric}'")

    def _load_cache(self) -> Dict[str, Dict[str, object]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            cached = data.get("data")
            if isinstance(cached, np.ndarray) and cached.size == 1:
                stored = cached.item()
                if isinstance(stored, dict):
                    return stored
        except Exception as exc:  # noqa: BLE001 - cache corruption fallback
            logger.warning("Failed to load embedding cache %s: %s", self.cache_path, exc)
        return {}

    def _save_cache(self, cache: Dict[str, Dict[str, object]]) -> None:
        try:
            np.savez(self.cache_path, data=cache)
        except Exception as exc:  # noqa: BLE001 - cache write failure
            logger.warning("Failed to persist embedding cache %s: %s", self.cache_path, exc)


@dataclass
class PerceptualHashStrategy(SimilarityStrategy):
    """Lightweight perceptual hash comparison using difference hash."""

    hash_size: int = 32

    def __post_init__(self) -> None:
        self.name = "perceptual_hash"
        self._library_hashes: Dict[Path, np.ndarray] = {}

    def prime(self, library_paths: Sequence[Path]) -> None:  # noqa: D401 - see base
        for path in library_paths:
            try:
                self._library_hashes[path] = self._hash(path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Hashing failed for %s: %s", path, exc)

    def find_matches(self, candidate: Path, *, top_k: int) -> List[StrategyMatch]:
        candidate_hash = self._hash(candidate)
        matches: List[StrategyMatch] = []

        for reference, ref_hash in self._library_hashes.items():
            similarity = self._hash_similarity(candidate_hash, ref_hash)
            matches.append(
                StrategyMatch(
                    candidate=candidate,
                    reference=reference,
                    score=similarity,
                    strategy=self.name,
                    reason=f"perceptual hash similarity {similarity:.3f}",
                    extra={"hash_size": self.hash_size},
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hash(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            resized = gray.resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)
            data = np.asarray(resized, dtype=np.float32)
            diff = data[:, 1:] > data[:, :-1]
            return diff.flatten()

    def _hash_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            size = min(a.size, b.size)
            a = a[:size]
            b = b[:size]
        hamming = float(np.count_nonzero(a != b))
        return float(1.0 - (hamming / max(1.0, a.size)))


def build_strategies(config: SimilarityConfig) -> List[SimilarityStrategy]:
    """Factory for the configured set of strategies."""

    strategies: List[SimilarityStrategy] = []
    for name in config.strategies:
        if name == "embedding":
            cache_path = config.cache_dir / "embedding_index.npz"
            strategies.append(
                EmbeddingSimilarityStrategy(
                    config=config,
                    cache_path=cache_path,
                    metric=config.similarity_metric,
                )
            )
        elif name in {"phash", "perceptual_hash", "dhash"}:
            strategies.append(PerceptualHashStrategy())
        else:
            raise ValueError(f"Unknown similarity strategy '{name}'")
    return strategies

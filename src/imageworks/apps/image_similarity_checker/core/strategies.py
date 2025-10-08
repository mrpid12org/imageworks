"""Similarity strategies combining deterministic and embedding approaches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import hashlib
import re
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
        cache = self._load_cache_legacy()

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
                if self.config.enable_augment_pooling:
                    encoded = {}
                    for p in to_encode:
                        try:
                            vec = self._embed_with_augment_pooling(embedding_model, p)
                            encoded[p] = vec
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "Augment pooling embed failed for %s: %s", p, exc
                            )
                else:
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
            self._save_cache_legacy(cache)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def find_matches(self, candidate: Path, *, top_k: int) -> List[StrategyMatch]:
        embedding_model = self._get_model()
        try:
            if self.config.enable_augment_pooling:
                vector = EmbeddingModel._normalise(
                    self._embed_with_augment_pooling(embedding_model, candidate)
                )
            else:
                vector = EmbeddingModel._normalise(embedding_model.embed(candidate))
        except Exception as exc:  # noqa: BLE001 - ensure graceful degradation
            logger.error("Failed to embed candidate %s: %s", candidate, exc)
            return []
        # Per-item loop
        matches: List[StrategyMatch] = []
        for reference, ref_vector in self._library_vectors.items():
            score = self._compute_similarity(vector, ref_vector)
            matches.append(
                StrategyMatch(
                    candidate=candidate,
                    reference=reference,
                    score=score,
                    strategy=self.name,
                    reason=(
                        f"cosine similarity {score:.3f}"
                        if self.metric == "cosine"
                        else "embedding comparison"
                    ),
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

    def _load_cache_legacy(self) -> Dict[str, Dict[str, object]]:
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
            logger.warning(
                "Failed to load embedding cache %s: %s", self.cache_path, exc
            )
        return {}

    def _save_cache_legacy(self, cache: Dict[str, Dict[str, object]]) -> None:
        try:
            np.savez(self.cache_path, data=cache)
        except Exception as exc:  # noqa: BLE001 - cache write failure
            logger.warning(
                "Failed to persist embedding cache %s: %s", self.cache_path, exc
            )

    # ------------------------------------------------------------------
    # Augmentation pooling helpers
    # ------------------------------------------------------------------
    def _embed_with_augment_pooling(
        self, model: EmbeddingModel, image_path: Path
    ) -> np.ndarray:
        """Compute a pooled embedding across configured augmentations.

        Variants: original (always), optional grayscale, optional five-crop (TL, TR, BL, BR, center).
        Each variant is embedded independently, then L2-normalised, averaged, and re-normalised.
        """
        variants: List[Image.Image] = []
        with Image.open(image_path) as img:
            base = img.convert("RGB")
            variants.append(base.copy())

            if self.config.augment_grayscale:
                try:
                    gray = base.convert("L").convert("RGB")
                    variants.append(gray)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("grayscale augment failed for %s: %s", image_path, exc)

            if self.config.augment_five_crop:
                try:
                    w, h = base.size
                    short = max(1, min(w, h))
                    ratio = float(self.config.augment_five_crop_ratio)
                    crop_size = max(
                        32, min(short, int(short * max(0.1, min(1.0, ratio))))
                    )
                    # compute corners + center boxes
                    boxes = [
                        (0, 0, crop_size, crop_size),  # TL
                        (w - crop_size, 0, w, crop_size),  # TR
                        (0, h - crop_size, crop_size, h),  # BL
                        (w - crop_size, h - crop_size, w, h),  # BR
                        (
                            (w - crop_size) // 2,
                            (h - crop_size) // 2,
                            (w + crop_size) // 2,
                            (h + crop_size) // 2,
                        ),  # C
                    ]
                    for box in boxes:
                        # sanitize box within image bounds
                        left, top, right, bottom = box
                        left = max(0, min(left, w - 1))
                        top = max(0, min(top, h - 1))
                        right = max(left + 1, min(right, w))
                        bottom = max(top + 1, min(bottom, h))
                        crop = base.crop((left, top, right, bottom)).resize(
                            (base.width, base.height), Image.Resampling.BILINEAR
                        )
                        variants.append(crop)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("five-crop augment failed for %s: %s", image_path, exc)

        embeddings: List[np.ndarray] = []
        # Embed each variant: write to a temporary in-memory file by saving to a bytes buffer if needed.
        # We reuse the file path embedding by saving variants to a temporary PNG in memory when backends accept file paths only.
        # Since current backends consume file paths, persist to a temp file next to the image when necessary.
        # For simplicity and safety, round-trip via numpy arrays using model.embed on the original path if model requires paths.
        # Here, we fallback to saving to a temporary .png under the cache dir.
        tmp_dir = self.cache_path.parent / "_aug_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        for idx, im in enumerate(variants):
            tmp_path = tmp_dir / f"{stem}.__aug{idx}.png"
            try:
                im.save(tmp_path, format="PNG")
                vec = model.embed(tmp_path)
                embeddings.append(EmbeddingModel._normalise(vec))
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "embedding failed for variant %d of %s: %s", idx, image_path, exc
                )
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

        if not embeddings:
            raise EmbeddingError(f"No embeddings produced for {image_path}")
        pooled = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        return EmbeddingModel._normalise(pooled)


@dataclass
class PerceptualHashStrategy(SimilarityStrategy):
    """Lightweight perceptual hash comparison using difference hash."""

    hash_size: int = 32
    cache_path: Path | None = None

    def __post_init__(self) -> None:
        self.name = "perceptual_hash"
        self._library_hashes: Dict[Path, np.ndarray] = {}
        self._library_hashes_packed: np.ndarray | None = None  # shape (N, nbytes)
        self._library_refs: List[Path] = []
        # 0..255 popcount lookup for vectorized Hamming
        self._popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def prime(self, library_paths: Sequence[Path]) -> None:  # noqa: D401 - see base
        cache: Dict[str, Dict[str, object]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                stored = data.get("data")
                if (
                    isinstance(stored, np.ndarray)
                    and stored.size == 1
                    and isinstance(stored.item(), dict)
                ):
                    cache = stored.item()  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to load pHash cache %s: %s", self.cache_path, exc)

        updated = False
        for path in library_paths:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            key = str(path)
            entry = cache.get(key)
            if entry and entry.get("mtime") == mtime:
                packed_list = entry.get("hash_packed", [])  # list[int]
                try:
                    packed = np.asarray(packed_list, dtype=np.uint8)
                    self._library_hashes[path] = (
                        packed  # store packed; mark using packed convention
                    )
                except Exception:
                    # fallback recompute if corrupted
                    h = self._hash(path)
                    packed = np.packbits(h, axis=0)
                    self._library_hashes[path] = packed
                    cache[key] = {"hash_packed": packed.tolist(), "mtime": mtime}
                    updated = True
            else:
                try:
                    h = self._hash(path)
                    packed = np.packbits(h, axis=0)
                    self._library_hashes[path] = packed
                    cache[key] = {"hash_packed": packed.tolist(), "mtime": mtime}
                    updated = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Hashing failed for %s: %s", path, exc)

        # Persist cache if changes
        if updated and self.cache_path:
            try:
                np.savez(self.cache_path, data=cache)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to save pHash cache %s: %s", self.cache_path, exc)

        # Build stacked packed hashes for vectorized matching
        if self._library_hashes:
            self._library_refs = list(self._library_hashes.keys())
            self._library_hashes_packed = np.vstack(
                [self._library_hashes[p] for p in self._library_refs]
            )

    def find_matches(self, candidate: Path, *, top_k: int) -> List[StrategyMatch]:
        # Prepare candidate packed hash
        try:
            cand_hash = self._hash(candidate)
            cand_packed = np.packbits(cand_hash, axis=0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Candidate hashing failed for %s: %s", candidate, exc)
            return []

        if self._library_hashes_packed is None or not self._library_refs:
            return []

        # Vectorized Hamming distance using XOR + popcount LUT
        xor = np.bitwise_xor(
            self._library_hashes_packed, cand_packed.reshape(1, -1)
        )  # (N, nbytes)
        # Map bytes to popcount via LUT and sum
        dists = self._popcnt[xor].sum(axis=1).astype(np.float32)  # (N,)
        nbits = float(self.hash_size * self.hash_size)
        sims = 1.0 - (dists / nbits)

        # Top-k selection
        if top_k < len(sims):
            idx = np.argpartition(-sims, top_k - 1)[:top_k]
            # Sort those indices by score desc
            idx = idx[np.argsort(-sims[idx])]
        else:
            idx = np.argsort(-sims)

        matches: List[StrategyMatch] = []
        for i in idx.tolist():
            reference = self._library_refs[i]
            similarity = float(sims[i])
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
        return matches[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hash(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            resized = gray.resize(
                (self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS
            )
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
            # Build a backend/model-specific cache name so caches don't collide
            backend = (config.embedding_backend or "embedding").strip().lower()
            # Prefer the embedding model id when provided (falls back to explainer model)
            model_id = (config.embedding_model or config.model or "").strip()
            if model_id:
                short = model_id.split("/")[-1]
                short = re.sub(r"[^A-Za-z0-9._-]", "-", short)[:48]
                digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
                aug_token = ""
                if config.enable_augment_pooling:
                    aug_token = (
                        f"__augpool-v1_g-{int(bool(config.augment_grayscale))}"
                        f"_5c-{int(bool(config.augment_five_crop))}"
                        f"_r-{str(config.augment_five_crop_ratio).replace('.', '_')}"
                    )
                cache_file = (
                    f"embedding_index__{backend}__{short}__{digest}{aug_token}.npz"
                )
            else:
                aug_token = ""
                if config.enable_augment_pooling:
                    aug_token = (
                        f"__augpool-v1_g-{int(bool(config.augment_grayscale))}"
                        f"_5c-{int(bool(config.augment_five_crop))}"
                        f"_r-{str(config.augment_five_crop_ratio).replace('.', '_')}"
                    )
                cache_file = f"embedding_index__{backend}{aug_token}.npz"
            cache_path = config.cache_dir / cache_file
            strategies.append(
                EmbeddingSimilarityStrategy(
                    config=config,
                    cache_path=cache_path,
                    metric=config.similarity_metric,
                )
            )
        elif name in {"phash", "perceptual_hash", "dhash"}:
            # Cache file keyed by library root and hash size
            lib_id = str(config.library_root)
            digest = hashlib.sha1(lib_id.encode("utf-8")).hexdigest()[:8]
            cache_file = f"phash_index__size{32}__{digest}.npz"  # hash_size is fixed in this strategy for now
            cache_path = config.cache_dir / cache_file
            strategies.append(PerceptualHashStrategy(cache_path=cache_path))
        else:
            raise ValueError(f"Unknown similarity strategy '{name}'")
    return strategies

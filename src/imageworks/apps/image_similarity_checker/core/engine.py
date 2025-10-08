"""Core execution engine for the image similarity checker."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

from imageworks.model_loader.service import CapabilityError, select_model

from .config import SimilarityConfig
from .discovery import discover_images, discover_library
from .explainer import SimilarityExplainer, create_explainer
from .metadata import SimilarityMetadataWriter
from .models import CandidateSimilarity, SimilarityVerdict, StrategyMatch
from .strategies import SimilarityStrategy, build_strategies

logger = logging.getLogger(__name__)


@dataclass
class EngineContext:
    """Container for runtime data used by the engine."""

    config: SimilarityConfig
    candidate_images: Sequence[Path]
    library_images: Sequence[Path]
    strategies: Sequence[SimilarityStrategy]


class SimilarityEngine:
    """Coordinate discovery, strategy execution, and reporting."""

    def __init__(
        self,
        config: SimilarityConfig,
        *,
        strategies: Optional[Sequence[SimilarityStrategy]] = None,
        metadata_writer: Optional[SimilarityMetadataWriter] = None,
    ) -> None:
        self.config = self._resolve_model_via_loader(config)
        self._strategies = list(strategies or build_strategies(self.config))
        self._metadata_writer = metadata_writer
        self._explainer: Optional[SimilarityExplainer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> List[CandidateSimilarity]:
        """Execute similarity analysis for the configured candidates."""

        candidate_images = discover_images(
            list(self.config.candidates),
            recursive=self.config.recursive,
            extensions=self.config.image_extensions,
        )
        if not candidate_images:
            raise ValueError("No candidate images were found for analysis")

        if self.config.dry_run:
            logger.info("Running similarity checker in dry-run mode")
            return [
                CandidateSimilarity(
                    candidate=path,
                    verdict=SimilarityVerdict.PASS,
                    top_score=0.0,
                    matches=[],
                    thresholds={
                        "fail": self.config.fail_threshold,
                        "query": self.config.query_threshold,
                    },
                    metric=self.config.similarity_metric,
                    notes=["Dry-run mode: similarity scores not computed"],
                )
                for path in candidate_images
            ]

        library_images = discover_library(
            self.config.library_root,
            recursive=self.config.recursive,
            extensions=self.config.image_extensions,
            exclude=candidate_images,
        )
        if not library_images:
            logger.warning(
                "No images discovered in library root %s", self.config.library_root
            )

        logger.info(
            "similarity_engine_start",
            extra={
                "event_type": "similarity_start",
                "candidates": len(candidate_images),
                "library": len(library_images),
                "strategies": [strategy.name for strategy in self._strategies],
            },
        )

        context = EngineContext(
            config=self.config,
            candidate_images=candidate_images,
            library_images=library_images,
            strategies=self._strategies,
        )

        for strategy in context.strategies:
            try:
                strategy.prime(context.library_images)
            except Exception as exc:  # noqa: BLE001
                logger.error("Strategy %s failed during prime(): %s", strategy.name, exc)

        results: List[CandidateSimilarity] = []
        for candidate in context.candidate_images:
            matches = self._evaluate_candidate(candidate, context.strategies)
            aggregated = self._aggregate_matches(matches)
            top_score = aggregated[0].score if aggregated else 0.0
            verdict = self._classify(top_score)
            result = CandidateSimilarity(
                candidate=candidate,
                verdict=verdict,
                top_score=top_score,
                matches=aggregated,
                thresholds={
                    "fail": self.config.fail_threshold,
                    "query": self.config.query_threshold,
                },
                metric=self.config.similarity_metric,
            )
            if not aggregated:
                result.notes.append("No comparable images found in library")
            if verdict == SimilarityVerdict.QUERY:
                result.notes.append(
                    "Similarity score within query band; manual review recommended"
                )
            elif verdict == SimilarityVerdict.FAIL:
                result.notes.append(
                    "Similarity score exceeds fail threshold; duplicate likely"
                )

            results.append(result)

            if self.config.write_metadata:
                self._write_metadata(candidate, result)

            if self.config.generate_explanations:
                explanation = self._ensure_explainer().explain(result)
                if explanation:
                    result.notes.append(f"LLM rationale: {explanation}")

        return results

    def close(self) -> None:  # pragma: no cover - runtime cleanup
        if self._explainer is not None:
            self._explainer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_candidate(
        self, candidate: Path, strategies: Sequence[SimilarityStrategy]
    ) -> List[StrategyMatch]:
        matches: List[StrategyMatch] = []
        for strategy in strategies:
            try:
                matches.extend(strategy.find_matches(candidate, top_k=self.config.top_matches))
            except Exception as exc:  # noqa: BLE001
                logger.error("Strategy %s failed for %s: %s", strategy.name, candidate, exc)
        return matches

    def _aggregate_matches(self, matches: Sequence[StrategyMatch]) -> List[StrategyMatch]:
        if not matches:
            return []

        aggregated: dict[Path, dict[str, object]] = {}
        for match in matches:
            reference = match.reference.resolve()
            entry = aggregated.setdefault(
                reference,
                {
                    "reference": match.reference,
                    "candidate": match.candidate,
                    "score": 0.0,
                    "strategies": {},
                    "reasons": [],
                    "details": {},
                },
            )
            entry["score"] = max(float(entry["score"]), float(match.score))
            entry["strategies"][match.strategy] = float(match.score)
            if match.reason:
                entry["reasons"].append(f"{match.strategy}: {match.reason}")
            if match.extra:
                entry["details"][match.strategy] = match.extra

        aggregated_matches: List[StrategyMatch] = []
        for info in aggregated.values():
            reason = " | ".join(info["reasons"]) if info["reasons"] else "combined score"
            aggregated_matches.append(
                StrategyMatch(
                    candidate=info["candidate"],
                    reference=info["reference"],
                    score=float(info["score"]),
                    strategy="ensemble",
                    reason=reason,
                    extra={
                        "strategies": info["strategies"],
                        "details": info["details"],
                    },
                )
            )

        aggregated_matches.sort(key=lambda item: item.score, reverse=True)
        return aggregated_matches[: self.config.top_matches]

    def _classify(self, score: float) -> SimilarityVerdict:
        if score >= self.config.fail_threshold:
            return SimilarityVerdict.FAIL
        if score >= self.config.query_threshold:
            return SimilarityVerdict.QUERY
        return SimilarityVerdict.PASS

    def _write_metadata(self, candidate: Path, result: CandidateSimilarity) -> None:
        writer = self._metadata_writer
        if writer is None:
            writer = SimilarityMetadataWriter(
                backup_originals=self.config.backup_originals,
                overwrite_existing=self.config.overwrite_metadata,
            )
            self._metadata_writer = writer
        try:
            wrote = writer.write(candidate, result)
            logger.debug("Metadata %s for %s", "written" if wrote else "skipped", candidate)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write metadata for %s: %s", candidate, exc)

    def _ensure_explainer(self) -> SimilarityExplainer:
        if self._explainer is None:
            self._explainer = create_explainer(self.config)
        return self._explainer

    def _resolve_model_via_loader(self, config: SimilarityConfig) -> SimilarityConfig:
        if not (config.use_loader or config.registry_model):
            return config

        logical_name = config.registry_model or config.model
        try:
            selected = select_model(logical_name, require_capabilities=["vision", "embedding"])
        except CapabilityError as exc:  # pragma: no cover - integration path
            raise RuntimeError(
                f"Model '{logical_name}' is missing required capabilities: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - propagate context
            raise RuntimeError(f"Failed to resolve model '{logical_name}': {exc}") from exc

        logger.info(
            "model_resolution",
            extra={
                "event_type": "model_resolution",
                "logical": logical_name,
                "endpoint": selected.endpoint_url,
                "internal_model": selected.internal_model_id,
                "backend": selected.backend,
            },
        )

        return replace(
            config,
            backend=selected.backend,
            base_url=selected.endpoint_url,
            model=selected.internal_model_id,
        )

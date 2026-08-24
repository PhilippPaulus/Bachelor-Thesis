from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from postbound import TableReference

from core.domain import EstimationResult, InternalPredicate, QualifiedTableName, TableArtifacts
from core.logging_utils import get_logger, log_kv
from integration.postbound.qal_utils import qualified_table_name
from model.naru.inference import NaruTableEstimator
from model.naru.persistence import artifact_paths, load_artifacts

logger = get_logger(__name__)


@dataclass(slots=True)
class RegistryEntry:
    artifacts: TableArtifacts
    estimator: NaruTableEstimator | None = None


@dataclass(slots=True)
class ModelRegistry:
    base_dir: Path
    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    bare_name_index: dict[str, list[str]] = field(default_factory=dict)
    inference_seed: int | None = None
    sample_count: int | None = None
    estimate_cache: dict[tuple[Any, ...], EstimationResult] = field(default_factory=dict)

    @classmethod
    def load(cls, base_dir: str | Path) -> "ModelRegistry":
        base = Path(base_dir).expanduser().resolve()
        entries: dict[str, RegistryEntry] = {}
        bare_name_index: dict[str, list[str]] = {}
        if not base.exists():
            return cls(base_dir=base, entries=entries, bare_name_index=bare_name_index)
        for schema_dir in base.iterdir():
            if not schema_dir.is_dir():
                continue
            for table_dir in schema_dir.iterdir():
                if not table_dir.is_dir():
                    continue
                artifacts = artifact_paths(base, schema_dir.name, table_dir.name)
                if not artifacts.is_complete():
                    continue
                try:
                    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
                    qualified_name = str(metadata["qualified_name"])
                except Exception:
                    continue
                entries[qualified_name] = RegistryEntry(artifacts=artifacts)
                bare_name_index.setdefault(table_dir.name, []).append(qualified_name)
        return cls(base_dir=base, entries=entries, bare_name_index=bare_name_index)

    def configure_inference(self, *, random_seed: int, sample_count: int | None = None) -> None:
        self.inference_seed = int(random_seed)
        self.sample_count = sample_count
        self.estimate_cache.clear()
        for entry in self.entries.values():
            if entry.estimator is not None:
                entry.estimator.configure_evaluation(
                    estimation_seed=self.inference_seed,
                    sample_count=self.sample_count,
                )

    def _resolve_key(self, table_name: str | TableReference, schema_name: str | None = None) -> str:
        if isinstance(table_name, TableReference):
            return qualified_table_name(table_name)
        try:
            return QualifiedTableName.parse(table_name, default_schema=schema_name).qualified_name
        except ValueError:
            matches = self.bare_name_index.get(table_name, [])
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise KeyError(f"No model registered for table '{table_name}'")
            raise KeyError(
                f"Ambiguous model lookup for '{table_name}'. Use a schema-qualified table name."
            )

    def has_model(self, table_name: str | TableReference, *, schema_name: str | None = None) -> bool:
        try:
            resolved = self._resolve_key(table_name, schema_name)
        except KeyError:
            return False
        return resolved in self.entries

    def available_tables(self) -> list[str]:
        return sorted(self.entries)

    def get_model(self, table_name: str | TableReference, *, schema_name: str | None = None) -> NaruTableEstimator:
        resolved_name = self._resolve_key(table_name, schema_name)
        entry = self.entries[resolved_name]
        if entry.estimator is None:
            model, encoder, config = load_artifacts(entry.artifacts)
            entry.estimator = NaruTableEstimator.create(
                model,
                encoder,
                config,
                estimation_seed=self.inference_seed,
                sample_count=self.sample_count,
            )
            log_kv(logger, "Loaded model lazily", table=resolved_name)
        return entry.estimator

    def estimate(
        self,
        table_name: str | TableReference,
        predicates: list[InternalPredicate],
        *,
        schema_name: str | None = None,
    ) -> EstimationResult:
        resolved_name = self._resolve_key(table_name, schema_name)
        cache_key = (
            resolved_name,
            self.inference_seed,
            self.sample_count,
            tuple(
                (
                    predicate.column_name,
                    predicate.operator.value,
                    repr(predicate.value),
                    repr(predicate.lower),
                    repr(predicate.upper),
                    tuple(repr(value) for value in predicate.values),
                )
                for predicate in predicates
            ),
        )
        if cache_key not in self.estimate_cache:
            estimator = self.get_model(resolved_name)
            self.estimate_cache[cache_key] = estimator.estimate(predicates)
        return self.estimate_cache[cache_key]

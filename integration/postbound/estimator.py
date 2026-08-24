from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from postbound import CardinalityEstimator as PostboundCardinalityEstimatorBase
from postbound import TableReference, qal as pb_qal
import postbound.transform as pb_transform

from core.domain import EstimationResult, InternalPredicate
from core.logging_utils import get_logger, log_kv
from registry.registry import ModelRegistry
from .qal_utils import build_table_query
from .translator import base_filter_predicate, extract_request_tables, translate_request

logger = get_logger(__name__)


@dataclass(slots=True)
class PostboundCardinalityEstimator(PostboundCardinalityEstimatorBase):
    registry: ModelRegistry

    def estimate_table(
        self,
        table_name: str,
        predicates: list[InternalPredicate],
        *,
        query: pb_qal.SqlQuery | None = None,
        intermediate: TableReference | Sequence[TableReference] | None = None,
    ) -> EstimationResult:
        if not self.registry.has_model(table_name):
            if query is None or intermediate is None:
                raise KeyError(f"No model registered for table '{table_name}'")
            return self._fallback(table_name, query, intermediate, "no trained model")
        try:
            result = self.registry.estimate(table_name, predicates)
        except Exception as exc:
            if query is None or intermediate is None:
                raise
            return self._fallback(table_name, query, intermediate, f"model inference failed: {exc}")
        log_kv(logger, "Naru estimate used", table=table_name, cardinality=result.cardinality)
        return result

    def calculate_estimate(
        self,
        query: pb_qal.SqlQuery,
        intermediate: TableReference | Sequence[TableReference] | None,
    ) -> float:
        return float(self.estimate_request(query, intermediate).cardinality)

    def initialize(self, target_db: Any, query: pb_qal.SqlQuery) -> None:  # pragma: no cover - lifecycle hook
        self.target_db = target_db
        self.query = query

    def cleanup(self) -> None:  # pragma: no cover - lifecycle hook
        return None

    def estimate_request(
        self,
        query: pb_qal.SqlQuery,
        intermediate: TableReference | Sequence[TableReference] | None,
    ) -> EstimationResult:
        translation = translate_request(query, intermediate)
        if not translation.can_estimate or translation.table_name is None:
            return self._fallback(None, query, intermediate, translation.fallback_reason or "translation failed")
        return self.estimate_table(
            translation.table_name,
            translation.predicates,
            query=query,
            intermediate=intermediate,
        )

    def _fallback(
        self,
        table_name: str | None,
        query: pb_qal.SqlQuery,
        intermediate: TableReference | Sequence[TableReference] | None,
        reason: str,
    ) -> EstimationResult:
        log_kv(logger, "Fallback used", table=table_name, reason=reason)
        fallback_value = self._native_fallback_value(query, intermediate)
        return EstimationResult(
            table_name=table_name or "unknown",
            selectivity=0.0,
            cardinality=float(fallback_value),
            used_model=False,
            reason=reason,
        )

    def _native_fallback_value(
        self,
        query: pb_qal.SqlQuery,
        intermediate: TableReference | Sequence[TableReference] | None,
    ) -> float:
        target_db = getattr(self, "target_db", None)
        if target_db is None:
            raise RuntimeError("PostBOUND estimator fallback requires initialize(target_db, query) before use")
        optimizer = target_db.optimizer()
        native_query = self._native_intermediate_query(query, intermediate)
        if native_query is not None:
            return float(optimizer.cardinality_estimate(native_query))
        return float(optimizer.cardinality_estimate(query))

    def _native_intermediate_query(
        self,
        query: pb_qal.SqlQuery,
        intermediate: TableReference | Sequence[TableReference] | None,
    ) -> pb_qal.SqlQuery | None:
        tables = extract_request_tables(intermediate)
        if not tables:
            return None
        if len(tables) != 1:
            query_tables = set(query.tables())
            if not set(tables).issubset(query_tables):
                return None
            fragment = pb_transform.extract_query_fragment(query, tables)
            return None if fragment is None else pb_transform.as_star_query(fragment)
        table = tables[0]
        predicate = base_filter_predicate(query, table)
        return build_table_query(table, predicate=predicate)

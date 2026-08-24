from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import postbound as pb
import postbound.transform as pb_transform
from postbound import TableReference, qal as pb_qal

from integration.postbound.estimator import PostboundCardinalityEstimator
from integration.postbound.qal_utils import build_table_query
from integration.postbound.translator import base_filter_predicate
from registry.registry import ModelRegistry

CardinalityConfig = Literal["native", "learned_base", "exact_base"]


@dataclass(frozen=True, slots=True)
class BaseEstimateRecord:
    table: str
    cardinality: float
    injected_cardinality: float
    used_model: bool
    fallback_reason: str | None = None
    model: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExactCardinalityMeasurement:
    sql: str
    cardinality: float
    status: Literal["executed", "cached"]
    duration_ms: float


@dataclass(slots=True)
class ExactCardinalityCache:
    database: Any
    values: dict[str, float] = field(default_factory=dict)

    def estimate_base(self, query: pb_qal.SqlQuery, table: TableReference) -> float:
        predicate = base_filter_predicate(query, table)
        count_query = build_table_query(table, predicate=predicate, count_star=True)
        key = str(count_query)
        return self._measure(count_query).cardinality

    def estimate_intermediate(
        self,
        query: pb_qal.SqlQuery,
        tables: frozenset[TableReference] | set[TableReference],
    ) -> float:
        return self.measure_intermediate(query, tables).cardinality

    def measure_intermediate(
        self,
        query: pb_qal.SqlQuery,
        tables: frozenset[TableReference] | set[TableReference],
    ) -> ExactCardinalityMeasurement:
        return self._measure(count_query_for_tables(query, tables))

    def _measure(self, count_query: pb_qal.SqlQuery) -> ExactCardinalityMeasurement:
        sql = str(count_query)
        if sql in self.values:
            return ExactCardinalityMeasurement(sql, self.values[sql], "cached", 0.0)
        started = time.perf_counter()
        value = _extract_scalar(self.database.execute_query(count_query, cache_enabled=False))
        duration_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid exact cardinality {value!r} for {sql}")
        self.values[sql] = value
        return ExactCardinalityMeasurement(sql, value, "executed", duration_ms)


class BaseOnlyCardinalityParameterization(pb.ParameterGeneration):
    def __init__(
        self,
        config: CardinalityConfig,
        database: Any,
        *,
        registry: ModelRegistry | None = None,
        exact_cache: ExactCardinalityCache | None = None,
    ) -> None:
        if config not in {"learned_base", "exact_base"}:
            raise ValueError("BaseOnlyCardinalityParameterization only supports learned_base and exact_base")
        self.config = config
        self.database = database
        self.registry = registry
        self.exact_cache = exact_cache or ExactCardinalityCache(database)
        self.records: list[BaseEstimateRecord] = []

    def generate_plan_parameters(
        self,
        query: pb_qal.SqlQuery,
        join_order: Any,
        operator_assignment: Any,
    ) -> pb.PlanParameterization:
        del join_order, operator_assignment
        self.records = []
        parameterization = pb.PlanParameterization()
        for table in sorted(query.tables(), key=table_key):
            record = self._estimate_table(query, table)
            self.records.append(record)
            if not math.isfinite(record.cardinality) or record.cardinality < 0:
                raise ValueError(f"Invalid {self.config} estimate for {record.table}: {record.cardinality!r}")
            parameterization.add_cardinality([table], record.injected_cardinality)
        return parameterization

    def describe(self) -> dict[str, Any]:
        return {"name": "base_only_cardinality_parameterization", "config": self.config}

    def _estimate_table(self, query: pb_qal.SqlQuery, table: TableReference) -> BaseEstimateRecord:
        if self.config == "exact_base":
            return BaseEstimateRecord(
                table=table_key(table),
                cardinality=(cardinality := self.exact_cache.estimate_base(query, table)),
                injected_cardinality=_injection_cardinality(cardinality),
                used_model=False,
                fallback_reason=None,
                model="exact_count",
            )
        if self.registry is None:
            raise ValueError("learned_base parameterization requires a model registry")
        estimator = PostboundCardinalityEstimator(registry=self.registry)
        estimator.initialize(self.database, query)
        result = estimator.estimate_request(query, table)
        return BaseEstimateRecord(
            table=table_key(table),
            cardinality=float(result.cardinality),
            injected_cardinality=_injection_cardinality(float(result.cardinality)),
            used_model=bool(result.used_model),
            fallback_reason=result.reason,
            model=result.table_name if result.used_model else None,
            diagnostics=dict(result.diagnostics),
        )


def cardinality_target_query(query: pb_qal.SqlQuery) -> pb_qal.SqlQuery:
    tables = sorted(query.tables(), key=table_key)
    if not tables:
        return pb_transform.as_star_query(query)
    fragment = pb_transform.extract_query_fragment(query, tables)
    return pb_transform.as_star_query(fragment or query)


def count_query_for_tables(
    query: pb_qal.SqlQuery,
    tables: frozenset[TableReference] | set[TableReference] | list[TableReference],
) -> pb_qal.SqlQuery:
    fragment = pb_transform.extract_query_fragment(query, sorted(tables, key=table_key))
    if fragment is None:
        raise ValueError(f"Could not extract query fragment for tables: {canonical_join_key(tables)}")
    return pb_transform.as_count_star_query(fragment)


def optimized_query_for_config(
    database: Any,
    query: pb_qal.SqlQuery,
    config: CardinalityConfig,
    *,
    registry: ModelRegistry | None = None,
    exact_cache: ExactCardinalityCache | None = None,
) -> tuple[pb_qal.SqlQuery, list[BaseEstimateRecord]]:
    if config == "native":
        return query, []
    if not hasattr(database, "hinting"):
        raise RuntimeError("Database does not expose PostBOUND hinting support")
    parameterization = BaseOnlyCardinalityParameterization(
        config,
        database,
        registry=registry,
        exact_cache=exact_cache,
    )
    pipeline = pb.MultiStageOptimizationPipeline(database)
    pipeline.setup_plan_parameterization(parameterization).build()
    optimized_query = pipeline.optimize_query(query)

    formatted_sql = str(optimized_query)
    if "/*=pg_lab=" not in formatted_sql or "Card(" not in formatted_sql:
        raise RuntimeError(
            "Expected project-generated pg_lab Card(...) hints, but the generated "
            f"{config} query did not contain them:\n{formatted_sql}"
        )

    return optimized_query, list(parameterization.records)


def final_cardinality_estimate(
    database: Any,
    query: pb_qal.SqlQuery,
    config: CardinalityConfig,
    *,
    registry: ModelRegistry | None = None,
    exact_cache: ExactCardinalityCache | None = None,
) -> tuple[float, list[BaseEstimateRecord]]:
    target_query = cardinality_target_query(query)
    optimized_query, records = optimized_query_for_config(
        database,
        target_query,
        config,
        registry=registry,
        exact_cache=exact_cache,
    )
    return float(database.optimizer().cardinality_estimate(optimized_query)), records


def query_plan_for_config(
    database: Any,
    query: pb_qal.SqlQuery,
    config: CardinalityConfig,
    *,
    registry: ModelRegistry | None = None,
    exact_cache: ExactCardinalityCache | None = None,
) -> tuple[pb.QueryPlan, pb_qal.SqlQuery, list[BaseEstimateRecord]]:
    optimized_query, records = optimized_query_for_config(
        database,
        query,
        config,
        registry=registry,
        exact_cache=exact_cache,
    )
    return database.optimizer().query_plan(optimized_query), optimized_query, records


def first_base_join(plan: pb.QueryPlan) -> frozenset[TableReference] | None:
    for node in plan.iternodes():
        if node.is_base_join():
            return frozenset(node.tables())
    return None


def first_base_join_key(plan: pb.QueryPlan) -> str | None:
    join = first_base_join(plan)
    if join is None:
        return None
    return canonical_join_key(join)


def table_key(table: TableReference) -> str:
    base_name = table.qualified_name() if table.schema else table.full_name
    alias = getattr(table, "alias", "")
    return f"{base_name}:{alias}" if alias else base_name


def canonical_join_key(tables: Any) -> str:
    return "|".join(sorted(table_key(table) for table in tables))


def records_to_json(records: list[BaseEstimateRecord]) -> str:
    return json.dumps(
        {record.table: record.cardinality for record in records},
        sort_keys=True,
    )


def injected_records_to_json(records: list[BaseEstimateRecord]) -> str:
    return json.dumps(
        {record.table: record.injected_cardinality for record in records},
        sort_keys=True,
    )


def ratio_json(
    numerator: dict[str, float],
    denominator: dict[str, float],
) -> str:
    ratios: dict[str, float | None] = {}
    for key, value in numerator.items():
        denom = denominator.get(key)
        ratios[key] = None if denom in (None, 0.0) else float(value) / float(denom)
    return json.dumps(ratios, sort_keys=True)


def base_estimates(
    database: Any,
    query: pb_qal.SqlQuery,
    config: CardinalityConfig,
    *,
    registry: ModelRegistry | None = None,
    exact_cache: ExactCardinalityCache | None = None,
) -> dict[str, float]:
    if config == "native":
        return {
            table_key(table): native_base_estimate(database, query, table)
            for table in sorted(query.tables(), key=table_key)
        }
    _, records = optimized_query_for_config(
        database,
        query,
        config,
        registry=registry,
        exact_cache=exact_cache,
    )
    return {record.table: record.cardinality for record in records}


def native_base_estimate(database: Any, query: pb_qal.SqlQuery, table: TableReference) -> float:
    predicate = base_filter_predicate(query, table)
    native_query = build_table_query(table, predicate=predicate)
    return float(database.optimizer().cardinality_estimate(native_query))


def _extract_scalar(result: Any) -> float:
    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("COUNT query returned no rows")
        first_row = result[0]
        if isinstance(first_row, (list, tuple)):
            return float(first_row[0])
        return float(first_row)
    return float(result)


def _injection_cardinality(value: float) -> float:
    """Returns PostgreSQL's representable estimate floor while preserving the raw value separately."""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"Cardinality must be finite and non-negative, got {value!r}")
    return max(numeric, 1.0)

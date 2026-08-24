from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.stats_ceb.cardinality import (
    BaseEstimateRecord,
    CardinalityConfig,
    ExactCardinalityCache,
    base_estimates,
    optimized_query_for_config,
)
from evaluation.stats_ceb.plans import (
    all_base_join_keys,
    explain_json,
    extract_hint,
    extract_joins,
    first_base_join_key,
    root_plan_rows,
    save_plan,
    save_sql,
    stable_plan_hash,
)
from evaluation.stats_ceb.preflight import validate_records_roundtrip
from evaluation.stats_ceb.run_context import RunContext


@dataclass(slots=True)
class TreatmentResult:
    config: CardinalityConfig
    query: Any
    sql: str
    hint: str
    document: dict[str, Any]
    plan_hash: str
    root_plan_rows: float
    first_base_join: str | None
    all_base_joins: list[str]
    all_joins: list[dict[str, Any]]
    base_estimates: dict[str, float]
    injected_base_estimates: dict[str, float]
    records: list[BaseEstimateRecord]
    sql_path: str
    plan_path: str
    hint_syntax_valid: bool
    hint_roundtrip_valid: bool
    treatment_valid: bool
    relation_roundtrips: list[dict[str, Any]]


def evaluate_treatments(
    context: RunContext,
    database: Any,
    registry: Any,
    exact_cache: ExactCardinalityCache,
    query: Any,
    query_id: str,
    *,
    configs: tuple[CardinalityConfig, ...] = ("native", "learned_base", "exact_base"),
    injection_validation: bool,
    timeout_seconds: float,
) -> dict[CardinalityConfig, TreatmentResult]:
    results: dict[CardinalityConfig, TreatmentResult] = {}
    for config in configs:
        optimized, records = optimized_query_for_config(
            database,
            query,
            config,
            registry=registry,
            exact_cache=exact_cache,
        )
        sql = str(optimized)
        hint = extract_hint(sql)
        hint_syntax_valid = (not hint) if config == "native" else ("/*=pg_lab=" in hint and "Card(" in hint)
        if not hint_syntax_valid:
            raise ValueError(f"Invalid {config} hint syntax for {query_id}: {sql}")
        document = explain_json(database, optimized, timeout_seconds=timeout_seconds)
        plan_path = context.run_dir / "plans" / query_id / f"{config}.json"
        sql_path = context.run_dir / "sql" / query_id / f"{config}.sql"
        save_plan(plan_path, document)
        save_sql(sql_path, optimized)
        relation_roundtrips: list[dict[str, Any]] = []
        if config != "native" and injection_validation:
            relation_roundtrips = validate_records_roundtrip(
                database,
                query,
                records,
                timeout_seconds=timeout_seconds,
            )
        roundtrip_valid = config == "native" or not injection_validation or all(
            bool(row["hint_roundtrip_valid"]) for row in relation_roundtrips
        )
        raw_estimates = (
            base_estimates(database, query, "native")
            if config == "native"
            else {record.table: record.cardinality for record in records}
        )
        injected = (
            dict(raw_estimates)
            if config == "native"
            else {record.table: record.injected_cardinality for record in records}
        )
        _assert_finite_estimates(raw_estimates, config)
        results[config] = TreatmentResult(
            config=config,
            query=optimized,
            sql=sql,
            hint=hint,
            document=document,
            plan_hash=stable_plan_hash(document),
            root_plan_rows=root_plan_rows(document),
            first_base_join=first_base_join_key(document),
            all_base_joins=all_base_join_keys(document),
            all_joins=extract_joins(document),
            base_estimates=raw_estimates,
            injected_base_estimates=injected,
            records=records,
            sql_path=context.relative(sql_path),
            plan_path=context.relative(plan_path),
            hint_syntax_valid=hint_syntax_valid,
            hint_roundtrip_valid=roundtrip_valid,
            treatment_valid=hint_syntax_valid and roundtrip_valid,
            relation_roundtrips=relation_roundtrips,
        )
    return results


def treatment_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def treatment_fields(result: TreatmentResult, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_sql": result.sql,
        f"{prefix}_hint": result.hint,
        f"{prefix}_plan_hash": result.plan_hash,
        f"{prefix}_plan_path": result.plan_path,
        f"{prefix}_sql_path": result.sql_path,
        f"{prefix}_first_base_join": result.first_base_join,
        f"{prefix}_all_base_joins": treatment_json(result.all_base_joins),
        f"{prefix}_all_joins": treatment_json(result.all_joins),
        f"{prefix}_base_estimates": treatment_json(result.base_estimates),
        f"{prefix}_injected_base_estimates": treatment_json(result.injected_base_estimates),
        f"{prefix}_hint_syntax_valid": result.hint_syntax_valid,
        f"{prefix}_hint_roundtrip_valid": result.hint_roundtrip_valid,
        f"{prefix}_treatment_valid": result.treatment_valid,
    }


def _assert_finite_estimates(estimates: dict[str, float], config: str) -> None:
    for key, value in estimates.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid {config} base estimate for {key}: {value!r}")

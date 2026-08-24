from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from postbound import TableReference

from evaluation.stats_ceb.cardinality import ExactCardinalityCache, native_base_estimate
from evaluation.stats_ceb.metrics import (
    DEFAULT_EQUALITY_TOLERANCE,
    paired_outcomes,
    q_error,
    signed_error_ratio,
    summarize_estimator,
)
from evaluation.stats_ceb.workload import StatsCebBaseQuery, StatsCebQuery, normalize_sql, normalized_sql_id
from integration.postbound.estimator import PostboundCardinalityEstimator
from integration.postbound.translator import translate_request
from registry.registry import ModelRegistry


def evaluate_single_table_workload(
    database: Any,
    registry: ModelRegistry,
    workload: list[StatsCebBaseQuery] | list[StatsCebQuery],
    *,
    exact_cache: ExactCardinalityCache | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_cache = exact_cache or ExactCardinalityCache(database)
    estimator = PostboundCardinalityEstimator(registry=registry)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    row_counts = registry_row_counts(registry)
    nullability_cache: dict[str, dict[str, bool]] = {}

    for raw_item in workload:
        item = _coerce_base_query(raw_item)
        tables = list(item.query.tables())
        base_row = _identity_row(item)
        if len(tables) != 1:
            error = f"query is not single-table (found {len(tables)} tables)"
            rows.append({**base_row, "status": "invalid", "error": error})
            failures.append(_failure(item, "single_table_validation", error))
            continue
        table = tables[0]
        started = time.perf_counter()
        try:
            actual = exact_cache.estimate_base(item.query, table)
            estimator.initialize(database, item.query)
            learned = estimator.estimate_request(item.query, table)
            native = native_base_estimate(database, item.query, table)
            translation = translate_request(item.query, table)
            if not translation.can_estimate:
                raise ValueError(translation.fallback_reason or "predicate translation failed")
            table_name = translation.table_name or item.table_name
            row_count = row_counts.get(table_name) or row_counts.get(table.full_name)
            if row_count is None or row_count <= 0:
                raise ValueError(f"Missing positive table row count for {table_name}")
            selectivity = float(actual) / float(row_count)
            predicate_count, operator_class, operators = predicate_shape(translation.predicates)
            columns = sorted({predicate.column_name for predicate in translation.predicates})
            nullability = nullability_cache.setdefault(
                table_name,
                _load_nullability(database, table),
            )
            nullable_columns = sorted(column for column in columns if nullability.get(column, False))
            diagnostics = dict(learned.diagnostics)
            constrained = diagnostics.get("constrained_columns", {})
            encodings = {
                column: payload.get("encoding_type")
                for column, payload in constrained.items()
                if isinstance(payload, dict)
            }
            learned_raw = float(learned.cardinality)
            native_raw = float(native)
            _assert_estimate(learned_raw, "learned")
            _assert_estimate(native_raw, "native")
            learned_ratio = signed_error_ratio(learned_raw, actual)
            native_ratio = signed_error_ratio(native_raw, actual)
            rows.append(
                {
                    **base_row,
                    "predicate_count": predicate_count,
                    "involved_columns": json.dumps(columns),
                    "operator_classes": json.dumps(sorted(set(_operator_class(op) for op in operators))),
                    "operator_class": operator_class,
                    "operators": json.dumps(operators),
                    "has_equality_predicate": any(op == "=" for op in operators),
                    "has_range_predicate": any(op in {"<", "<=", ">", ">=", "between"} for op in operators),
                    "has_null_sensitive_column": bool(nullable_columns),
                    "nullable_constrained_columns": json.dumps(nullable_columns),
                    "encoding_types": json.dumps(encodings, sort_keys=True),
                    "exact_cardinality": actual,
                    "actual": actual,
                    "table_cardinality": row_count,
                    "table_row_count": row_count,
                    "selectivity": selectivity,
                    "native_estimate": native_raw,
                    "learned_estimate": learned_raw,
                    "learned_raw_estimate": learned_raw,
                    "native_metric_estimate": max(native_raw, 1.0),
                    "learned_metric_estimate": max(learned_raw, 1.0),
                    "model_used": learned.table_name if learned.used_model else None,
                    "used_model": bool(learned.used_model),
                    "fallback_used": not learned.used_model,
                    "fallback_reason": learned.reason,
                    "inference_mode": diagnostics.get("inference_mode", "fallback"),
                    "sample_count": diagnostics.get("sample_count", 0),
                    "estimator_seed": diagnostics.get("estimator_seed"),
                    "native_q_error": q_error(native_raw, actual),
                    "learned_q_error": q_error(learned_raw, actual),
                    "native_signed_error_ratio": native_ratio,
                    "learned_signed_error_ratio": learned_ratio,
                    "learned_overestimation": learned_ratio > 1.0 + DEFAULT_EQUALITY_TOLERANCE,
                    "learned_underestimation": learned_ratio < 1.0 - DEFAULT_EQUALITY_TOLERANCE,
                    "estimate_direction": estimate_direction(learned_raw, actual),
                    "estimate_difference_ratio_learned_native": None
                    if native_raw == 0
                    else learned_raw / native_raw,
                    "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
                    "processing_duration_ms": (time.perf_counter() - started) * 1000.0,
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:
            error = str(exc)
            rows.append(
                {
                    **base_row,
                    "processing_duration_ms": (time.perf_counter() - started) * 1000.0,
                    "status": "error",
                    "error": error,
                }
            )
            failures.append(_failure(item, "single_table_evaluation", error))
    return rows, failures


def single_table_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    unique = _unique_rows(valid)
    filtered = [row for row in valid if bool(row["is_filtered"])]
    unique_filtered = _unique_rows(filtered)
    unfiltered = [row for row in valid if not bool(row["is_filtered"])]
    return {
        "row_count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "model_used_count": sum(bool(row["used_model"]) for row in valid),
        "model_fallback_count": sum(bool(row["fallback_used"]) for row in valid),
        "sections": {
            "all_workload_occurrences": _section(valid),
            "unique_normalized_sql": _section(unique),
            "filtered_workload_occurrences": _section(filtered),
            "unique_filtered_sql": _section(unique_filtered),
            "unfiltered_full_table_cases": _section(unfiltered),
        },
        "definitions": {
            "q_error": "max(max(estimate, 1)/max(exact, 1), max(exact, 1)/max(estimate, 1))",
            "signed_error_ratio": "max(estimate, 1)/max(exact, 1)",
            "sub_unit_policy": "raw estimate retained; only metric inputs and pg_lab injection use floor 1",
            "equality_tolerance": DEFAULT_EQUALITY_TOLERANCE,
        },
    }


def predicate_shape(predicates: list[Any]) -> tuple[int, str, list[str]]:
    operators = [str(predicate.operator.value) for predicate in predicates]
    if not operators:
        return 0, "no_filter", []
    classes = {_operator_class(operator) for operator in operators}
    return (len(operators), next(iter(classes)), operators) if len(classes) == 1 else (len(operators), "mixed", operators)


def estimate_direction(estimate: float, actual: float) -> str:
    ratio = signed_error_ratio(estimate, actual)
    if ratio > 1.0 + DEFAULT_EQUALITY_TOLERANCE:
        return "over"
    if ratio < 1.0 - DEFAULT_EQUALITY_TOLERANCE:
        return "under"
    return "equal"


def registry_row_counts(registry: ModelRegistry) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for qualified_name, entry in registry.entries.items():
        try:
            metadata = json.loads(Path(entry.artifacts.metadata_path).read_text(encoding="utf-8"))
            row_count = int(metadata["row_count"])
        except Exception:
            continue
        row_counts[qualified_name] = row_count
        row_counts[qualified_name.split(".", 1)[-1]] = row_count
    return row_counts


def _operator_class(operator: str) -> str:
    if operator == "=":
        return "equality"
    if operator in {"<", "<=", ">", ">="}:
        return "range"
    if operator in {"between", "in"}:
        return operator
    return "other"


def _coerce_base_query(item: StatsCebBaseQuery | StatsCebQuery) -> StatsCebBaseQuery:
    if isinstance(item, StatsCebBaseQuery):
        return item
    tables = list(item.query.tables())
    table = tables[0] if len(tables) == 1 else TableReference("invalid")
    table_name = table.qualified_name() if table.schema else table.full_name
    alias = str(getattr(table, "alias", "") or "")
    relation_key = f"{table_name}:{alias}" if alias else table_name
    predicate = None if len(tables) != 1 else item.query.predicates()
    return StatsCebBaseQuery(
        occurrence_id=item.label,
        normalized_sql_id=normalized_sql_id(item.query),
        normalized_sql=normalize_sql(item.query),
        original_query_label=item.label,
        original_query_id=item.query_id,
        original_sql=item.sql,
        template=item.template,
        table_name=table_name,
        table_alias=alias,
        table_key=relation_key,
        predicate_text="" if predicate is None else str(predicate),
        query=item.query,
        sql=item.sql,
        is_filtered=predicate is not None,
    )


def _identity_row(item: StatsCebBaseQuery) -> dict[str, Any]:
    return {
        "occurrence_id": item.occurrence_id,
        "label": item.occurrence_id,
        "normalized_sql_id": item.normalized_sql_id,
        "normalized_sql": item.normalized_sql,
        "original_query_id": item.original_query_label,
        "source_query_id": item.original_query_id,
        "query_id": item.original_query_id,
        "query_template": item.template,
        "template": item.template,
        "sql": item.sql,
        "original_complete_sql": item.original_sql,
        "table": item.table_name,
        "table_alias": item.table_alias,
        "table_key": item.table_key,
        "predicate_text": item.predicate_text,
        "is_filtered": item.is_filtered,
    }


def _failure(item: StatsCebBaseQuery, stage: str, reason: str) -> dict[str, Any]:
    return {"label": item.occurrence_id, "stage": stage, "reason": reason, "sql": item.sql}


def _section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    learned = [float(row["learned_q_error"]) for row in rows]
    native = [float(row["native_q_error"]) for row in rows]
    outcomes = paired_outcomes(learned, native)
    learned_summary = summarize_estimator(rows, "learned")
    native_summary = summarize_estimator(rows, "native")
    return {
        "count": len(rows),
        "unique_query_count": len({row["normalized_sql_id"] for row in rows}),
        "learned": learned_summary,
        "native": native_summary,
        "paired_outcomes": {
            "learned_better": outcomes["left_better"],
            "native_better": outcomes["right_better"],
            "equal": outcomes["equal"],
        },
        "paired_differences": None
        if not rows
        else {
            "median_q_error_learned_minus_native": statistics.median(learned) - statistics.median(native),
            "geometric_mean_q_error_learned_minus_native": math.exp(statistics.fmean(math.log(x) for x in learned))
            - math.exp(statistics.fmean(math.log(x) for x in native)),
            "fraction_q_error_le2_learned_minus_native": sum(x <= 2 for x in learned) / len(learned)
            - sum(x <= 2 for x in native) / len(native),
        },
    }


def _unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(str(row["normalized_sql_id"]), row)
    return list(by_id.values())


def _load_nullability(database: Any, table: TableReference) -> dict[str, bool]:
    schema = str(table.schema or "public").replace("'", "''")
    name = str(table.full_name).replace("'", "''")
    sql = (
        "SELECT column_name, is_nullable FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}'"
    )
    result = database.execute_query(sql, cache_enabled=False, raw=True)
    return {str(column).lower(): str(nullable).upper() == "YES" for column, nullable in result}


def _assert_estimate(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} estimate is not finite and non-negative: {value!r}")


SINGLE_TABLE_FIELDS = [
    "occurrence_id",
    "normalized_sql_id",
    "normalized_sql",
    "original_query_id",
    "source_query_id",
    "query_template",
    "sql",
    "original_complete_sql",
    "table",
    "table_alias",
    "table_key",
    "predicate_text",
    "is_filtered",
    "predicate_count",
    "involved_columns",
    "operator_classes",
    "operator_class",
    "operators",
    "has_equality_predicate",
    "has_range_predicate",
    "has_null_sensitive_column",
    "nullable_constrained_columns",
    "encoding_types",
    "exact_cardinality",
    "table_cardinality",
    "selectivity",
    "native_estimate",
    "learned_estimate",
    "learned_raw_estimate",
    "native_metric_estimate",
    "learned_metric_estimate",
    "model_used",
    "fallback_used",
    "fallback_reason",
    "inference_mode",
    "sample_count",
    "estimator_seed",
    "native_q_error",
    "learned_q_error",
    "native_signed_error_ratio",
    "learned_signed_error_ratio",
    "learned_overestimation",
    "learned_underestimation",
    "estimate_direction",
    "estimate_difference_ratio_learned_native",
    "diagnostics_json",
    "processing_duration_ms",
    "status",
    "error",
]

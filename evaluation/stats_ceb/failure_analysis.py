from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from evaluation.stats_ceb.metrics import summarize


SYSTEMATIC_MIN_GROUP_SIZE = 5
SYSTEMATIC_MEDIAN_MULTIPLIER = 2.0


def cardinality_bucket(value: float) -> str:
    if value <= 0:
        return "exactly_zero"
    if value <= 1:
        return "(0,1]"
    if value <= 10:
        return "(1,10]"
    if value <= 100:
        return "(10,100]"
    if value <= 1_000:
        return "(100,1k]"
    if value <= 10_000:
        return "(1k,10k]"
    if value <= 100_000:
        return "(10k,100k]"
    return ">100k"


def selectivity_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value == 0:
        return "exactly_zero"
    for boundary in (0.0001, 0.001, 0.01, 0.1, 0.5, 1.0):
        if value <= boundary:
            lower = 0 if boundary == 0.0001 else {0.001: 0.0001, 0.01: 0.001, 0.1: 0.01, 0.5: 0.1, 1.0: 0.5}[boundary]
            return f"({lower:g},{boundary:g}]"
    return ">1"


def build_category_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status", "ok") == "ok"]
    overall = summarize([float(row["learned_q_error"]) for row in valid]) or {}
    overall_median = float(overall.get("median", 0.0))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        for category_name, category_value in _categories(row):
            grouped[(category_name, category_value)].append(row)

    output: list[dict[str, Any]] = []
    for (category_name, category_value), group_rows in sorted(grouped.items()):
        learned = [float(row["learned_q_error"]) for row in group_rows]
        native = [float(row["native_q_error"]) for row in group_rows]
        learned_summary = summarize(learned) or {}
        native_summary = summarize(native) or {}
        median = float(learned_summary.get("median", 0.0))
        systematic = len(group_rows) >= SYSTEMATIC_MIN_GROUP_SIZE and (
            overall_median > 0 and median >= SYSTEMATIC_MEDIAN_MULTIPLIER * overall_median
        )
        output.append(
            {
                "group_dimension": category_name,
                "group_value": category_value,
                "group_size": len(group_rows),
                "unique_query_count": len({row["normalized_sql_id"] for row in group_rows}),
                "learned_median_q_error": learned_summary.get("median"),
                "learned_geometric_mean_q_error": learned_summary.get("geometric_mean"),
                "learned_p90_q_error": learned_summary.get("p90"),
                "learned_p95_q_error": learned_summary.get("p95"),
                "learned_p99_q_error": learned_summary.get("p99"),
                "learned_max_q_error": learned_summary.get("max"),
                "learned_gt10_percent": learned_summary.get("gt10_percent"),
                "learned_gt100_percent": learned_summary.get("gt100_percent"),
                "learned_overestimation_percent": 100.0
                * sum(_as_bool(row["learned_overestimation"]) for row in group_rows)
                / len(group_rows),
                "learned_median_signed_ratio": _median(
                    float(row["learned_signed_error_ratio"]) for row in group_rows
                ),
                "native_median_q_error": native_summary.get("median"),
                "native_geometric_mean_q_error": native_summary.get("geometric_mean"),
                "native_p95_q_error": native_summary.get("p95"),
                "native_gt10_percent": native_summary.get("gt10_percent"),
                "systematic_failure_candidate": systematic,
                "systematic_min_group_size": SYSTEMATIC_MIN_GROUP_SIZE,
                "systematic_median_multiplier": SYSTEMATIC_MEDIAN_MULTIPLIER,
                "overall_learned_median_q_error": overall_median,
            }
        )
    return output


def worst_queries(rows: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status", "ok") == "ok"]
    systematic_groups = {
        (row["group_dimension"], row["group_value"])
        for row in build_category_metrics(valid)
        if bool(row["systematic_failure_candidate"])
    }
    by_q_error = sorted(valid, key=lambda row: float(row["learned_q_error"]), reverse=True)[:limit]
    by_disadvantage = sorted(
        valid,
        key=lambda row: float(row["learned_q_error"]) / float(row["native_q_error"]),
        reverse=True,
    )[:limit]
    selected: dict[str, dict[str, Any]] = {}
    q_error_ids = {str(row["occurrence_id"]) for row in by_q_error}
    disadvantage_ids = {str(row["occurrence_id"]) for row in by_disadvantage}
    for row in [*by_q_error, *by_disadvantage]:
        occurrence = str(row["occurrence_id"])
        payload = dict(row)
        payload["learned_native_q_error_ratio"] = float(row["learned_q_error"]) / float(row["native_q_error"])
        payload["worst_learned_q_error"] = occurrence in q_error_ids
        payload["worst_learned_native_disadvantage"] = occurrence in disadvantage_ids
        matching_systematic = [
            {"dimension": dimension, "value": value}
            for dimension, value in _categories(row)
            if (dimension, value) in systematic_groups
        ]
        candidate, evidence = _candidate_classification(row, matching_systematic)
        payload["candidate_classification"] = candidate
        payload["classification_status"] = "candidate_requires_manual_review"
        payload["classification_evidence"] = json.dumps(evidence, sort_keys=True)
        selected[occurrence] = payload
    return sorted(selected.values(), key=lambda row: float(row["learned_q_error"]), reverse=True)


def top_failures(rows: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    return worst_queries(rows, limit=limit)


def _categories(row: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "table", str(row["table"])
    yield "query_template", str(row["query_template"])
    yield "predicate_count", str(row["predicate_count"])
    yield "operator_class", str(row["operator_class"])
    yield "equality_range_shape", _predicate_shape(row)
    yield "constrained_column_count", str(len(_json_list(row.get("involved_columns"))))
    yield "exact_cardinality_bucket", cardinality_bucket(float(row["exact_cardinality"]))
    yield "selectivity_bucket", selectivity_bucket(_optional_float(row.get("selectivity")))
    yield "estimate_direction", str(row["estimate_direction"])
    yield "inference_mode", str(row["inference_mode"])
    yield "nullable_constrained_columns", "nullable" if _as_bool(row["has_null_sensitive_column"]) else "non_nullable"
    encodings = _json_dict(row.get("encoding_types"))
    for column in _json_list(row.get("involved_columns")):
        yield "constrained_column", f"{row['table']}.{column}"
        yield "encoding_type", str(encodings.get(column, "unknown"))


def _predicate_shape(row: dict[str, Any]) -> str:
    equality = _as_bool(row.get("has_equality_predicate"))
    range_predicate = _as_bool(row.get("has_range_predicate"))
    if equality and range_predicate:
        return "mixed"
    if equality:
        return "equality"
    if range_predicate:
        return "range"
    return "other_or_unfiltered"


def _candidate_classification(
    row: dict[str, Any], matching_systematic: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    evidence = {
        "learned_q_error": float(row["learned_q_error"]),
        "native_q_error": float(row["native_q_error"]),
        "inference_mode": row.get("inference_mode"),
        "fallback_used": _as_bool(row.get("fallback_used")),
        "diagnostics_available": bool(row.get("diagnostics_json")),
        "repeated_estimation_variance_available": False,
        "matching_systematic_groups": matching_systematic,
    }
    if matching_systematic:
        return "systematic category failure", evidence
    if _as_bool(row.get("fallback_used")):
        return "preprocessing/translation issue", evidence
    if row.get("inference_mode") == "progressive_sampling":
        return "sampling instability", evidence
    if float(row["learned_q_error"]) > 10 and float(row["native_q_error"]) < float(row["learned_q_error"]):
        return "likely model limitation", evidence
    return "isolated outlier", evidence


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


CATEGORY_FIELDS = [
    "group_dimension", "group_value", "group_size", "unique_query_count",
    "learned_median_q_error", "learned_geometric_mean_q_error", "learned_p90_q_error",
    "learned_p95_q_error", "learned_p99_q_error", "learned_max_q_error",
    "learned_gt10_percent", "learned_gt100_percent", "learned_overestimation_percent",
    "learned_median_signed_ratio", "native_median_q_error", "native_geometric_mean_q_error",
    "native_p95_q_error", "native_gt10_percent", "systematic_failure_candidate",
    "systematic_min_group_size", "systematic_median_multiplier", "overall_learned_median_q_error",
]

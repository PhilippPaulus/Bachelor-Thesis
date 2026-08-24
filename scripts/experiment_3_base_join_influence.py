from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cardinality import ExactCardinalityCache, cardinality_target_query
from evaluation.stats_ceb.cli import add_common_db_model_args, load_database_and_registry, prepare_run_context, select_workload
from evaluation.stats_ceb.metrics import q_error, wilson_interval
from evaluation.stats_ceb.preflight import ensure_preflight, select_preflight_query
from evaluation.stats_ceb.reports import write_csv, write_json
from evaluation.stats_ceb.run_context import build_manifest, write_or_update_manifest
from evaluation.stats_ceb.treatments import evaluate_treatments, treatment_json
from evaluation.stats_ceb.workload import join_count, load_stats_ceb_workload


FIELDS = [
    "query_id",
    "source_query_id",
    "query_template",
    "relation_count",
    "join_count",
    "native_first_base_join",
    "learned_first_base_join",
    "exact_base_first_base_join",
    "native_all_base_joins",
    "learned_all_base_joins",
    "exact_base_all_base_joins",
    "base_join_changed_native_learned",
    "full_plan_changed_native_learned",
    "native_plan_hash",
    "learned_plan_hash",
    "exact_base_plan_hash",
    "native_base_estimates",
    "learned_base_estimates",
    "exact_base_estimates",
    "learned_native_ratios",
    "maximum_absolute_log_cardinality_difference",
    "native_final_q_error",
    "learned_final_q_error",
    "exact_base_final_q_error",
    "single_table_estimate_outcome",
    "native_hint",
    "learned_hint",
    "exact_base_hint",
    "native_plan_path",
    "learned_plan_path",
    "exact_base_plan_path",
    "preflight_passed",
    "hint_syntax_valid",
    "hint_roundtrip_valid",
    "treatment_valid",
    "applicable",
    "status",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3: influence of learned base estimates on base joins")
    add_common_db_model_args(parser)
    parser.add_argument("--complete-workload-path", required=True)
    parser.add_argument("--single-table-results-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workload = select_workload(
        load_stats_ceb_workload(
            args.complete_workload_path,
            workload_format="complete",
            template_map_path=args.template_map,
        ),
        args,
    )
    context = prepare_run_context(args)
    output_dir = context.experiment_dir("experiment_3")
    if args.resume and (output_dir / "summary.json").exists():
        print(f"Experiment 3 already complete: {output_dir}")
        return
    database, registry = load_database_and_registry(args)
    write_or_update_manifest(
        context,
        build_manifest(
            context,
            database=database,
            registry=registry,
            args=args,
            workload_sources=[args.complete_workload_path],
            workload_query_count=len(workload),
        ),
    )
    preflight = ensure_preflight(
        context,
        database=database,
        registry=registry,
        sample_query=_preflight_query(workload),
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
    )
    single_outcomes = _load_single_table_outcomes(
        Path(args.single_table_results_path).expanduser().resolve()
        if args.single_table_results_path
        else context.run_dir / "experiment_1_accuracy" / "single_table_results.csv"
    )
    rows = evaluate_workload(
        context,
        database,
        registry,
        workload,
        preflight=preflight,
        single_outcomes=single_outcomes,
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
    )
    write_csv(output_dir / "base_join_changes.csv", rows, FIELDS)
    write_json(output_dir / "summary.json", build_summary(rows))


def evaluate_workload(
    context: Any,
    database: Any,
    registry: Any,
    workload: list[Any],
    *,
    preflight: dict[str, Any],
    single_outcomes: dict[str, str],
    injection_validation: bool,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    cache = ExactCardinalityCache(database)
    rows: list[dict[str, Any]] = []
    for item in workload:
        target = cardinality_target_query(item.query)
        treatments = evaluate_treatments(
            context,
            database,
            registry,
            cache,
            target,
            item.label,
            injection_validation=injection_validation,
            timeout_seconds=timeout_seconds,
        )
        native, learned, exact = treatments["native"], treatments["learned_base"], treatments["exact_base"]
        applicable = all(result.first_base_join is not None for result in treatments.values())
        actual = float(item.actual_cardinality) if item.actual_cardinality is not None else None
        if actual is None:
            raise ValueError(f"Missing exact final cardinality for {item.label}")
        ratios = {
            key: None if native.base_estimates.get(key) == 0 else value / native.base_estimates[key]
            for key, value in learned.base_estimates.items()
            if key in native.base_estimates
        }
        log_differences = [
            abs(math.log(max(float(learned.base_estimates[key]), 1.0) / max(float(native.base_estimates[key]), 1.0)))
            for key in learned.base_estimates
            if key in native.base_estimates
        ]
        rows.append(
            {
                "query_id": item.label,
                "source_query_id": item.query_id,
                "query_template": item.template,
                "relation_count": item.query_size,
                "join_count": join_count(item.query),
                "native_first_base_join": native.first_base_join,
                "learned_first_base_join": learned.first_base_join,
                "exact_base_first_base_join": exact.first_base_join,
                "native_all_base_joins": treatment_json(native.all_base_joins),
                "learned_all_base_joins": treatment_json(learned.all_base_joins),
                "exact_base_all_base_joins": treatment_json(exact.all_base_joins),
                "base_join_changed_native_learned": applicable and native.first_base_join != learned.first_base_join,
                "full_plan_changed_native_learned": native.plan_hash != learned.plan_hash,
                "native_plan_hash": native.plan_hash,
                "learned_plan_hash": learned.plan_hash,
                "exact_base_plan_hash": exact.plan_hash,
                "native_base_estimates": treatment_json(native.base_estimates),
                "learned_base_estimates": treatment_json(learned.base_estimates),
                "exact_base_estimates": treatment_json(exact.base_estimates),
                "learned_native_ratios": treatment_json(ratios),
                "maximum_absolute_log_cardinality_difference": max(log_differences, default=0.0),
                "native_final_q_error": q_error(native.root_plan_rows, actual),
                "learned_final_q_error": q_error(learned.root_plan_rows, actual),
                "exact_base_final_q_error": q_error(exact.root_plan_rows, actual),
                "single_table_estimate_outcome": single_outcomes.get(item.label, "unknown"),
                "native_hint": native.hint,
                "learned_hint": learned.hint,
                "exact_base_hint": exact.hint,
                "native_plan_path": native.plan_path,
                "learned_plan_path": learned.plan_path,
                "exact_base_plan_path": exact.plan_path,
                "preflight_passed": preflight["preflight_passed"],
                "hint_syntax_valid": learned.hint_syntax_valid and exact.hint_syntax_valid,
                "hint_roundtrip_valid": learned.hint_roundtrip_valid and exact.hint_roundtrip_valid,
                "treatment_valid": all(result.treatment_valid for result in treatments.values()),
                "applicable": applicable,
                "status": "ok" if applicable else "not_applicable",
                "error": None if applicable else "one or more plans contain no base join",
            }
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if _as_bool(row["applicable"]) and _as_bool(row["treatment_valid"])]
    changed = sum(_as_bool(row["base_join_changed_native_learned"]) for row in applicable)
    full_changed = sum(_as_bool(row["full_plan_changed_native_learned"]) for row in applicable)
    groups: dict[str, dict[str, Any]] = {}
    dimensions = {
        "relation_count": lambda row: str(row["relation_count"]),
        "query_template": lambda row: str(row["query_template"]),
        "estimate_divergence": lambda row: _divergence_bucket(float(row["maximum_absolute_log_cardinality_difference"])),
        "single_table_estimate_outcome": lambda row: str(row["single_table_estimate_outcome"]),
    }
    for dimension, key_fn in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in applicable:
            grouped[key_fn(row)].append(row)
        groups[dimension] = {
            key: {
                **wilson_interval(
                    sum(_as_bool(row["base_join_changed_native_learned"]) for row in group),
                    len(group),
                )
            }
            for key, group in sorted(grouped.items())
        }
    change_interval = wilson_interval(changed, len(applicable))
    return {
        "experiment": "base_join_influence",
        "total_query_count": len(rows),
        "applicable_query_count": len(applicable),
        "invalid_or_not_applicable_count": len(rows) - len(applicable),
        "changed_base_join": change_interval,
        "changed_base_join_percent": None if not applicable else 100.0 * changed / len(applicable),
        "full_plan_change": wilson_interval(full_changed, len(applicable)),
        "practical_relevance_threshold": 0.05,
        "practical_relevance_threshold_met": bool(
            change_interval["rate"] is not None and float(change_interval["rate"]) >= 0.05
        ),
        "groups": groups,
        "interpretation_limit": "This experiment establishes influence only; changed joins are not labeled better or worse.",
    }


def _load_single_table_outcomes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            grouped[str(row["original_query_id"])].append(
                (float(row["learned_q_error"]), float(row["native_q_error"]))
            )
    outcomes: dict[str, str] = {}
    for query_id, pairs in grouped.items():
        learned = sorted(pair[0] for pair in pairs)
        native = sorted(pair[1] for pair in pairs)
        learned_median = statistics.median(learned)
        native_median = statistics.median(native)
        outcomes[query_id] = "improved" if learned_median < native_median else (
            "worsened" if learned_median > native_median else "unchanged"
        )
    return outcomes


def _divergence_bucket(value: float) -> str:
    if value < math.log(1.5):
        return "<1.5x"
    if value < math.log(2):
        return "1.5x-2x"
    if value < math.log(10):
        return "2x-10x"
    if value < math.log(100):
        return "10x-100x"
    return ">=100x"


def _preflight_query(workload: list[Any]) -> Any:
    return select_preflight_query(workload)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


if __name__ == "__main__":
    main()

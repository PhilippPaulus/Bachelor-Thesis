from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cardinality import ExactCardinalityCache, cardinality_target_query
from evaluation.stats_ceb.cli import (
    add_common_db_model_args,
    load_database_and_registry,
    prepare_run_context,
    select_workload,
)
from evaluation.stats_ceb.metrics import (
    DEFAULT_EQUALITY_TOLERANCE,
    geometric_mean,
    paired_bootstrap_ci,
    paired_cluster_bootstrap_ci,
    paired_outcomes,
    proportion_le2,
    q_error,
    summarize,
)
from evaluation.stats_ceb.preflight import ensure_preflight, select_preflight_query
from evaluation.stats_ceb.reports import write_csv, write_failures, write_json
from evaluation.stats_ceb.run_context import build_manifest, write_or_update_manifest
from evaluation.stats_ceb.single_table import (
    SINGLE_TABLE_FIELDS,
    evaluate_single_table_workload,
    single_table_summary,
)
from evaluation.stats_ceb.treatments import evaluate_treatments, treatment_fields, treatment_json
from evaluation.stats_ceb.workload import derive_base_queries, join_count, load_stats_ceb_workload


COMPLETE_FIELDS = [
    "query_id",
    "source_query_id",
    "query_template",
    "relation_count",
    "join_count",
    "original_count_sql",
    "aggregate_free_sql",
    "exact_complete_query_cardinality",
    "native_final_estimate",
    "learned_base_final_estimate",
    "exact_base_final_estimate",
    "native_q_error",
    "learned_base_q_error",
    "exact_base_q_error",
    "learned_versus_native_winner",
    "exact_base_versus_native_winner",
    "learned_improvement_ratio",
    "exact_base_attainable_improvement_ratio",
    "native_plan_hash",
    "learned_plan_hash",
    "exact_base_plan_hash",
    "native_first_base_join",
    "learned_first_base_join",
    "exact_base_first_base_join",
    "native_base_estimates",
    "learned_base_estimates",
    "exact_base_estimates",
    "learned_hint",
    "exact_base_hint",
    "native_plan_path",
    "learned_plan_path",
    "exact_base_plan_path",
    "preflight_passed",
    "hint_syntax_valid",
    "hint_roundtrip_valid",
    "treatment_valid",
    "status",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 1: STATS-CEB estimate accuracy and propagation")
    add_common_db_model_args(parser)
    parser.add_argument("--complete-workload-path", required=True)
    parser.add_argument(
        "--single-table-workload-path",
        default=None,
        help="Legacy provenance input; Experiment 1A derives base-table occurrences from the complete workload.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    complete_workload = select_workload(
        load_stats_ceb_workload(
            args.complete_workload_path,
            workload_format="complete",
            template_map_path=args.template_map,
        ),
        args,
    )
    context = prepare_run_context(args)
    output_dir = context.experiment_dir("experiment_1")
    if args.resume and (output_dir / "summary.json").exists():
        print(f"Experiment 1 already complete: {output_dir}")
        return
    database, registry = load_database_and_registry(args)
    workload_sources = [args.complete_workload_path]
    if args.single_table_workload_path:
        workload_sources.append(args.single_table_workload_path)
    write_or_update_manifest(
        context,
        build_manifest(
            context,
            database=database,
            registry=registry,
            args=args,
            workload_sources=workload_sources,
            workload_query_count=len(complete_workload),
        ),
    )
    preflight = ensure_preflight(
        context,
        database=database,
        registry=registry,
        sample_query=_preflight_query(complete_workload),
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
    )
    exact_cache = ExactCardinalityCache(database)

    base_occurrences = derive_base_queries(complete_workload)
    single_rows, failures = evaluate_single_table_workload(
        database,
        registry,
        base_occurrences,
        exact_cache=exact_cache,
    )
    complete_rows = evaluate_complete_workload(
        context,
        database,
        registry,
        complete_workload,
        exact_cache,
        preflight=preflight,
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
    )
    single_summary = single_table_summary(single_rows)
    complete_summary = build_complete_summary(complete_rows)
    statistics_payload = build_statistics(
        single_rows,
        complete_rows,
        samples=args.bootstrap_samples,
        random_seed=args.bootstrap_seed,
    )
    summary = {
        "experiment": "accuracy_and_propagation",
        "single_table": single_summary,
        "complete_query": complete_summary,
        "propagation_conclusion": _propagation_conclusion(single_summary, complete_summary),
        "invalid_single_table_count": len(failures),
    }
    write_csv(output_dir / "single_table_results.csv", single_rows, SINGLE_TABLE_FIELDS)
    write_csv(output_dir / "complete_query_results.csv", complete_rows, COMPLETE_FIELDS)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "statistics.json", statistics_payload)
    write_failures(output_dir, failures)


def evaluate_complete_workload(
    context: Any,
    database: Any,
    registry: Any,
    workload: list[Any],
    exact_cache: ExactCardinalityCache,
    *,
    preflight: dict[str, Any],
    injection_validation: bool,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in workload:
        if item.actual_cardinality is None:
            raise ValueError(f"Complete workload row {item.label} lacks an exact final cardinality")
        target = cardinality_target_query(item.query)
        if "count(" in str(target).lower():
            raise AssertionError(f"Aggregate removal failed for {item.label}")
        treatments = evaluate_treatments(
            context,
            database,
            registry,
            exact_cache,
            target,
            item.label,
            injection_validation=injection_validation,
            timeout_seconds=timeout_seconds,
        )
        native = treatments["native"]
        learned = treatments["learned_base"]
        exact_base = treatments["exact_base"]
        if not all(value.treatment_valid for value in treatments.values()):
            raise ValueError(f"Invalid optimizer treatment for {item.label}")
        actual = float(item.actual_cardinality)
        native_q = q_error(native.root_plan_rows, actual)
        learned_q = q_error(learned.root_plan_rows, actual)
        exact_q = q_error(exact_base.root_plan_rows, actual)
        row = {
            "query_id": item.label,
            "source_query_id": item.query_id,
            "query_template": item.template,
            "relation_count": item.query_size,
            "join_count": join_count(item.query),
            "original_count_sql": item.sql,
            "aggregate_free_sql": str(target),
            "exact_complete_query_cardinality": actual,
            "native_final_estimate": native.root_plan_rows,
            "learned_base_final_estimate": learned.root_plan_rows,
            "exact_base_final_estimate": exact_base.root_plan_rows,
            "native_q_error": native_q,
            "learned_base_q_error": learned_q,
            "exact_base_q_error": exact_q,
            "learned_versus_native_winner": _winner(learned_q, native_q, "learned_base", "native"),
            "exact_base_versus_native_winner": _winner(exact_q, native_q, "exact_base", "native"),
            "learned_improvement_ratio": native_q / learned_q,
            "exact_base_attainable_improvement_ratio": native_q / exact_q,
            "native_plan_hash": native.plan_hash,
            "learned_plan_hash": learned.plan_hash,
            "exact_base_plan_hash": exact_base.plan_hash,
            "native_first_base_join": native.first_base_join,
            "learned_first_base_join": learned.first_base_join,
            "exact_base_first_base_join": exact_base.first_base_join,
            "native_base_estimates": treatment_json(native.base_estimates),
            "learned_base_estimates": treatment_json(learned.base_estimates),
            "exact_base_estimates": treatment_json(exact_base.base_estimates),
            "learned_hint": learned.hint,
            "exact_base_hint": exact_base.hint,
            "native_plan_path": native.plan_path,
            "learned_plan_path": learned.plan_path,
            "exact_base_plan_path": exact_base.plan_path,
            "preflight_passed": bool(preflight["preflight_passed"]),
            "hint_syntax_valid": learned.hint_syntax_valid and exact_base.hint_syntax_valid,
            "hint_roundtrip_valid": learned.hint_roundtrip_valid and exact_base.hint_roundtrip_valid,
            "treatment_valid": learned.treatment_valid and exact_base.treatment_valid,
            "status": "ok",
            "error": None,
        }
        rows.append(row)
    return rows


def build_complete_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "ok" and bool(row["treatment_valid"])]
    native = [float(row["native_q_error"]) for row in valid]
    learned = [float(row["learned_base_q_error"]) for row in valid]
    exact = [float(row["exact_base_q_error"]) for row in valid]
    outcomes = paired_outcomes(learned, native)
    exact_outcomes = paired_outcomes(exact, native)
    return {
        "query_count": len(rows),
        "valid_query_count": len(valid),
        "invalid_query_count": len(rows) - len(valid),
        "native_q_error": summarize(native),
        "learned_base_q_error": summarize(learned),
        "exact_base_q_error": summarize(exact),
        "learned_versus_native": {
            "learned_better_count": outcomes["left_better"],
            "native_better_count": outcomes["right_better"],
            "unchanged_count": outcomes["equal"],
            "learned_better_share": None if not valid else outcomes["left_better"] / len(valid),
        },
        "exact_base_versus_native": {
            "exact_base_better_count": exact_outcomes["left_better"],
            "native_better_count": exact_outcomes["right_better"],
            "unchanged_count": exact_outcomes["equal"],
        },
        "median_learned_improvement_ratio": None
        if not valid
        else statistics.median(float(row["learned_improvement_ratio"]) for row in valid),
        "median_exact_base_attainable_improvement_ratio": None
        if not valid
        else statistics.median(float(row["exact_base_attainable_improvement_ratio"]) for row in valid),
    }


def build_statistics(
    single_rows: list[dict[str, Any]],
    complete_rows: list[dict[str, Any]],
    *,
    samples: int,
    random_seed: int,
) -> dict[str, Any]:
    single = [row for row in single_rows if row.get("status") == "ok"]
    complete = [row for row in complete_rows if row.get("status") == "ok"]
    return {
        "definitions": {
            "bootstrap_samples": samples,
            "bootstrap_seed": random_seed,
            "occurrence_bootstrap_unit": "workload base-table occurrence",
            "cluster_bootstrap_units": ["normalized_sql_id", "query_template"],
            "confidence_level": 0.95,
            "delta_direction": "learned minus native; negative Q-error deltas favor learned",
        },
        "single_table": _bootstrap_bundle(
            [float(row["learned_q_error"]) for row in single],
            [float(row["native_q_error"]) for row in single],
            [row["normalized_sql_id"] for row in single],
            [row["query_template"] for row in single],
            samples=samples,
            seed=random_seed,
        ),
        "complete_query": _bootstrap_bundle(
            [float(row["learned_base_q_error"]) for row in complete],
            [float(row["native_q_error"]) for row in complete],
            [row["query_id"] for row in complete],
            [row["query_template"] for row in complete],
            samples=samples,
            seed=random_seed,
        ),
    }


def _bootstrap_bundle(
    learned: list[float],
    native: list[float],
    normalized_clusters: list[str],
    template_clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Callable[[list[float]], float]] = {
        "median_q_error": statistics.median,
        "geometric_mean_q_error": lambda values: float(geometric_mean(values)),
        "fraction_q_error_le2": proportion_le2,
    }
    return {
        name: {
            "occurrence_weighted": paired_bootstrap_ci(
                learned,
                native,
                metric,
                samples=samples,
                random_seed=seed,
            ),
            "clustered_by_normalized_sql": paired_cluster_bootstrap_ci(
                learned,
                native,
                normalized_clusters,
                metric,
                samples=samples,
                random_seed=seed,
            ),
            "clustered_by_query_template": paired_cluster_bootstrap_ci(
                learned,
                native,
                template_clusters,
                metric,
                samples=samples,
                random_seed=seed,
            ),
        }
        for name, metric in metrics.items()
    }


def _winner(left: float, right: float, left_name: str, right_name: str) -> str:
    scale = max(left, right, 1.0)
    if abs(left - right) <= DEFAULT_EQUALITY_TOLERANCE * scale:
        return "unchanged"
    return left_name if left < right else right_name


def _preflight_query(workload: list[Any]) -> Any:
    return select_preflight_query(workload)


def _propagation_conclusion(single: dict[str, Any], complete: dict[str, Any]) -> dict[str, Any]:
    single_section = single["sections"]["filtered_workload_occurrences"]
    single_outcomes = single_section["paired_outcomes"]
    complete_outcomes = complete["learned_versus_native"]
    single_improved = single_outcomes["learned_better"] > single_outcomes["native_better"]
    final_improved = complete_outcomes["learned_better_count"] > complete_outcomes["native_better_count"]
    return {
        "single_table_estimates_improved": single_improved,
        "complete_query_estimates_improved": final_improved,
        "improvement_propagated": single_improved and final_improved,
        "interpretation": "Propagation is reported only when paired outcomes favor learned estimates at both levels.",
    }


if __name__ == "__main__":
    main()

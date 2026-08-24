from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cardinality import ExactCardinalityCache, cardinality_target_query, table_key
from evaluation.stats_ceb.cli import add_common_db_model_args, load_database_and_registry, prepare_run_context, select_workload
from evaluation.stats_ceb.metrics import geometric_mean, paired_bootstrap_ci, percentile, wilson_interval
from evaluation.stats_ceb.preflight import ensure_preflight, select_preflight_query
from evaluation.stats_ceb.reports import write_csv, write_json
from evaluation.stats_ceb.run_context import build_manifest, write_or_update_manifest
from evaluation.stats_ceb.treatments import evaluate_treatments
from evaluation.stats_ceb.workload import load_stats_ceb_workload


RELATIVE_OUTPUT_EPSILON = 1.0

FIELDS = [
    "query_id",
    "source_query_id",
    "query_template",
    "relation_count",
    "native_first_base_join",
    "learned_first_base_join",
    "exact_base_reference_first_join",
    "native_agrees_exact_base_reference",
    "learned_agrees_exact_base_reference",
    "decision_category",
    "native_exact_first_join_output",
    "learned_exact_first_join_output",
    "exact_base_reference_exact_first_join_output",
    "native_exact_join_sql",
    "learned_exact_join_sql",
    "exact_base_reference_exact_join_sql",
    "native_exact_cardinality_status",
    "learned_exact_cardinality_status",
    "exact_base_reference_exact_cardinality_status",
    "native_exact_cardinality_duration_ms",
    "learned_exact_cardinality_duration_ms",
    "exact_base_reference_exact_cardinality_duration_ms",
    "native_relative_first_join_output",
    "learned_relative_first_join_output",
    "native_plan_hash",
    "learned_plan_hash",
    "exact_base_plan_hash",
    "native_plan_path",
    "learned_plan_path",
    "exact_base_plan_path",
    "preflight_passed",
    "treatment_valid",
    "missing_value_status",
    "status",
    "error",
]

CACHE_FIELDS = [
    "query_id",
    "configuration",
    "join_key",
    "exact_cardinality",
    "count_sql",
    "cache_status",
    "duration_ms",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4: base-join decision quality against exact-base reference")
    add_common_db_model_args(parser)
    parser.add_argument("--complete-workload-path", required=True)
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
    output_dir = context.experiment_dir("experiment_4")
    if args.resume and (output_dir / "summary.json").exists():
        print(f"Experiment 4 already complete: {output_dir}")
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
    rows, cache_rows = evaluate_workload(
        context,
        database,
        registry,
        workload,
        preflight=preflight,
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
    )
    write_csv(output_dir / "decision_quality.csv", rows, FIELDS)
    write_csv(output_dir / "exact_join_cardinality_cache.csv", cache_rows, CACHE_FIELDS)
    write_json(
        output_dir / "summary.json",
        build_summary(rows, samples=args.bootstrap_samples, random_seed=args.bootstrap_seed),
    )


def evaluate_workload(
    context: Any,
    database: Any,
    registry: Any,
    workload: list[Any],
    *,
    preflight: dict[str, Any],
    injection_validation: bool,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache = ExactCardinalityCache(database)
    rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
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
        keys = {
            "native": native.first_base_join,
            "learned": learned.first_base_join,
            "exact_base_reference": exact.first_base_join,
        }
        if any(key is None for key in keys.values()):
            rows.append(
                {
                    "query_id": item.label,
                    "source_query_id": item.query_id,
                    "query_template": item.template,
                    "relation_count": item.query_size,
                    "native_first_base_join": native.first_base_join,
                    "learned_first_base_join": learned.first_base_join,
                    "exact_base_reference_first_join": exact.first_base_join,
                    "decision_category": "not applicable",
                    "native_plan_hash": native.plan_hash,
                    "learned_plan_hash": learned.plan_hash,
                    "exact_base_plan_hash": exact.plan_hash,
                    "native_plan_path": native.plan_path,
                    "learned_plan_path": learned.plan_path,
                    "exact_base_plan_path": exact.plan_path,
                    "preflight_passed": preflight["preflight_passed"],
                    "treatment_valid": all(result.treatment_valid for result in treatments.values()),
                    "missing_value_status": "base_join_not_available",
                    "status": "not_applicable",
                    "error": "one or more plans contain no base join",
                }
            )
            continue
        measurements: dict[str, Any] = {}
        for configuration, key in keys.items():
            assert key is not None
            tables = _tables_for_join(target, key)
            measurement = cache.measure_intermediate(target, tables)
            measurements[configuration] = measurement
            cache_rows.append(
                {
                    "query_id": item.label,
                    "configuration": configuration,
                    "join_key": key,
                    "exact_cardinality": measurement.cardinality,
                    "count_sql": measurement.sql,
                    "cache_status": measurement.status,
                    "duration_ms": measurement.duration_ms,
                }
            )
        reference_value = measurements["exact_base_reference"].cardinality
        native_relative = max(measurements["native"].cardinality, RELATIVE_OUTPUT_EPSILON) / max(
            reference_value, RELATIVE_OUTPUT_EPSILON
        )
        learned_relative = max(measurements["learned"].cardinality, RELATIVE_OUTPUT_EPSILON) / max(
            reference_value, RELATIVE_OUTPUT_EPSILON
        )
        native_agrees = keys["native"] == keys["exact_base_reference"]
        learned_agrees = keys["learned"] == keys["exact_base_reference"]
        rows.append(
            {
                "query_id": item.label,
                "source_query_id": item.query_id,
                "query_template": item.template,
                "relation_count": item.query_size,
                "native_first_base_join": keys["native"],
                "learned_first_base_join": keys["learned"],
                "exact_base_reference_first_join": keys["exact_base_reference"],
                "native_agrees_exact_base_reference": native_agrees,
                "learned_agrees_exact_base_reference": learned_agrees,
                "decision_category": classify_decision(keys["native"], keys["learned"], keys["exact_base_reference"]),
                "native_exact_first_join_output": measurements["native"].cardinality,
                "learned_exact_first_join_output": measurements["learned"].cardinality,
                "exact_base_reference_exact_first_join_output": reference_value,
                "native_exact_join_sql": measurements["native"].sql,
                "learned_exact_join_sql": measurements["learned"].sql,
                "exact_base_reference_exact_join_sql": measurements["exact_base_reference"].sql,
                "native_exact_cardinality_status": measurements["native"].status,
                "learned_exact_cardinality_status": measurements["learned"].status,
                "exact_base_reference_exact_cardinality_status": measurements["exact_base_reference"].status,
                "native_exact_cardinality_duration_ms": measurements["native"].duration_ms,
                "learned_exact_cardinality_duration_ms": measurements["learned"].duration_ms,
                "exact_base_reference_exact_cardinality_duration_ms": measurements["exact_base_reference"].duration_ms,
                "native_relative_first_join_output": native_relative,
                "learned_relative_first_join_output": learned_relative,
                "native_plan_hash": native.plan_hash,
                "learned_plan_hash": learned.plan_hash,
                "exact_base_plan_hash": exact.plan_hash,
                "native_plan_path": native.plan_path,
                "learned_plan_path": learned.plan_path,
                "exact_base_plan_path": exact.plan_path,
                "preflight_passed": preflight["preflight_passed"],
                "treatment_valid": all(result.treatment_valid for result in treatments.values()),
                "missing_value_status": "complete",
                "status": "ok",
                "error": None,
            }
        )
    return rows, cache_rows


def classify_decision(native_key: str | None, learned_key: str | None, exact_base_key: str | None) -> str:
    if native_key is None or learned_key is None or exact_base_key is None:
        return "not applicable"
    native_agrees = native_key == exact_base_key
    learned_agrees = learned_key == exact_base_key
    if learned_agrees and not native_agrees:
        return "improved"
    if native_agrees and not learned_agrees:
        return "degraded"
    if native_agrees and learned_agrees:
        return "both agree"
    return "neither agrees"


def build_summary(rows: list[dict[str, Any]], *, samples: int = 10_000, random_seed: int = 42) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and _as_bool(row["treatment_valid"])]
    native_agreement = [1.0 if _as_bool(row["native_agrees_exact_base_reference"]) else 0.0 for row in valid]
    learned_agreement = [1.0 if _as_bool(row["learned_agrees_exact_base_reference"]) else 0.0 for row in valid]
    native_relative = [float(row["native_relative_first_join_output"]) for row in valid]
    learned_relative = [float(row["learned_relative_first_join_output"]) for row in valid]
    categories = {
        category: sum(row["decision_category"] == category for row in valid)
        for category in ("improved", "degraded", "both agree", "neither agrees")
    }
    native_agreement_rate = None if not valid else sum(native_agreement) / len(valid)
    learned_agreement_rate = None if not valid else sum(learned_agreement) / len(valid)
    native_relative_geomean = geometric_mean(native_relative)
    learned_relative_geomean = geometric_mean(learned_relative)
    conclusion = bool(
        valid
        and learned_agreement_rate is not None
        and native_agreement_rate is not None
        and learned_agreement_rate > native_agreement_rate
        and learned_relative_geomean is not None
        and native_relative_geomean is not None
        and learned_relative_geomean < native_relative_geomean
        and categories["improved"] > categories["degraded"]
    )
    return {
        "experiment": "base_join_decision_quality",
        "reference_definition": (
            "The exact-base reference uses the same optimizer, join estimator, cost model, and search strategy, "
            "with exact filtered base-table cardinalities. It is not a globally optimal oracle."
        ),
        "query_count": len(rows),
        "valid_query_count": len(valid),
        "invalid_or_not_applicable_count": len(rows) - len(valid),
        "native_exact_base_agreement": wilson_interval(int(sum(native_agreement)), len(valid)),
        "learned_exact_base_agreement": wilson_interval(int(sum(learned_agreement)), len(valid)),
        "paired_agreement_difference_learned_minus_native": paired_bootstrap_ci(
            learned_agreement,
            native_agreement,
            statistics.fmean,
            samples=samples,
            random_seed=random_seed,
        ),
        "decision_categories": categories,
        "native_relative_first_join_output": _relative_summary(native_relative),
        "learned_relative_first_join_output": _relative_summary(learned_relative),
        "learned_smaller_exact_output_count": sum(l < n for l, n in zip(learned_relative, native_relative)),
        "native_smaller_exact_output_count": sum(n < l for l, n in zip(learned_relative, native_relative)),
        "paired_mean_log_relative_output_difference_learned_minus_native": paired_bootstrap_ci(
            [math.log(value) for value in learned_relative],
            [math.log(value) for value in native_relative],
            statistics.fmean,
            samples=samples,
            random_seed=random_seed,
        ),
        "relative_output_epsilon": RELATIVE_OUTPUT_EPSILON,
        "improvement_criteria": {
            "higher_exact_base_agreement_required": True,
            "lower_aggregate_relative_output_required": True,
            "improvements_must_outnumber_degradations": True,
        },
        "decision_quality_improved_under_all_criteria": conclusion,
    }


def _relative_summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "geometric_mean": geometric_mean(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def _tables_for_join(query: Any, join_key: str) -> frozenset[Any]:
    by_key = {table_key(table): table for table in query.tables()}
    requested = join_key.split("|")
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"Join key references tables not found in query: {missing}")
    tables = frozenset(by_key[key] for key in requested)
    if len(tables) != 2:
        raise ValueError(f"Expected a two-relation base join, got {join_key}")
    return tables


def _preflight_query(workload: list[Any]) -> Any:
    return select_preflight_query(workload)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


if __name__ == "__main__":
    main()

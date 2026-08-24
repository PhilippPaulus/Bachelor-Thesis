from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cardinality import CardinalityConfig, ExactCardinalityCache, cardinality_target_query, optimized_query_for_config
from evaluation.stats_ceb.cli import add_common_db_model_args, load_database_and_registry, prepare_run_context, select_workload
from evaluation.stats_ceb.metrics import geometric_mean, paired_bootstrap_ci, percentile
from evaluation.stats_ceb.plans import analyze_metrics, explain_json, extract_hint, save_plan
from evaluation.stats_ceb.preflight import ensure_preflight, select_preflight_query
from evaluation.stats_ceb.reports import write_csv, write_json
from evaluation.stats_ceb.run_context import build_manifest, write_or_update_manifest
from evaluation.stats_ceb.treatments import evaluate_treatments, treatment_json
from evaluation.stats_ceb.workload import load_stats_ceb_workload


PRACTICAL_RUNTIME_THRESHOLD = 0.05

REPETITION_FIELDS = [
    "query_id",
    "source_query_id",
    "query_template",
    "relation_count",
    "configuration",
    "repetition",
    "warmup",
    "execution_order",
    "execution_sql",
    "hint",
    "base_join",
    "base_join_changed",
    "plan_hash",
    "full_plan_changed",
    "treatment_category",
    "plan_path",
    "analyze_plan_path",
    "wall_clock_duration_ms",
    "planning_time_ms",
    "execution_time_ms",
    "timeout",
    "timeout_seconds",
    "shared_hit_blocks",
    "shared_read_blocks",
    "shared_dirtied_blocks",
    "shared_written_blocks",
    "temp_read_blocks",
    "temp_written_blocks",
    "preflight_passed",
    "treatment_valid",
    "status",
    "error",
]

PER_QUERY_FIELDS = [
    "query_id",
    "source_query_id",
    "query_template",
    "relation_count",
    "native_base_join",
    "learned_base_join",
    "base_join_changed",
    "native_plan_hash",
    "learned_plan_hash",
    "full_plan_changed",
    "treatment_category",
    "native_median_execution_time_ms",
    "learned_median_execution_time_ms",
    "native_median_wall_clock_ms",
    "learned_median_wall_clock_ms",
    "speedup_native_over_learned",
    "runtime_classification",
    "native_timeout_count",
    "learned_timeout_count",
    "valid_measured_repetitions",
    "status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5: end-to-end runtime impact")
    add_common_db_model_args(parser)
    parser.add_argument("--complete-workload-path", required=True)
    parser.add_argument("--include-exact-reference", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 5:
        raise ValueError("Runtime experiment requires at least five measured repetitions")
    workload = select_workload(
        load_stats_ceb_workload(
            args.complete_workload_path,
            workload_format="complete",
            template_map_path=args.template_map,
        ),
        args,
    )
    context = prepare_run_context(args)
    output_dir = context.experiment_dir("experiment_5")
    if args.resume and (output_dir / "summary.json").exists():
        print(f"Experiment 5 already complete: {output_dir}")
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
    configs: list[CardinalityConfig] = ["native", "learned_base"]
    if args.include_exact_reference:
        configs.append("exact_base")
    repetitions = evaluate_runtime(
        context,
        output_dir,
        database,
        registry,
        workload,
        configs=configs,
        preflight=preflight,
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
        warmups=args.warmups,
        measured_repetitions=args.repetitions,
        cache_policy=args.cache_policy,
    )
    per_query = build_per_query(repetitions, timeout_seconds=args.timeout_seconds)
    summary = build_summary(
        repetitions,
        per_query_rows=per_query,
        timeout_seconds=args.timeout_seconds,
        samples=args.bootstrap_samples,
        random_seed=args.bootstrap_seed,
    )
    write_csv(output_dir / "runtime_repetitions.csv", repetitions, REPETITION_FIELDS)
    write_csv(output_dir / "runtime_per_query.csv", per_query, PER_QUERY_FIELDS)
    write_json(output_dir / "summary.json", summary)


def evaluate_runtime(
    context: Any,
    output_dir: Path,
    database: Any,
    registry: Any,
    workload: list[Any],
    *,
    configs: list[CardinalityConfig],
    preflight: dict[str, Any],
    injection_validation: bool,
    timeout_seconds: float,
    warmups: int,
    measured_repetitions: int,
    cache_policy: str,
) -> list[dict[str, Any]]:
    cache = ExactCardinalityCache(database)
    rows: list[dict[str, Any]] = []
    for item in workload:
        target = cardinality_target_query(item.query)
        plan_treatments = evaluate_treatments(
            context,
            database,
            registry,
            cache,
            target,
            item.label,
            configs=tuple(configs),
            injection_validation=injection_validation,
            timeout_seconds=timeout_seconds,
        )
        native = plan_treatments["native"]
        learned = plan_treatments["learned_base"]
        execution_queries: dict[str, Any] = {}
        for config in configs:
            execution_query, _ = optimized_query_for_config(
                database,
                item.query,
                config,
                registry=registry,
                exact_cache=cache,
            )
            execution_queries[config] = execution_query
        base_changed = native.first_base_join != learned.first_base_join
        full_changed = native.plan_hash != learned.plan_hash
        treatment_category = classify_treatment(
            base_changed,
            full_changed,
            native.base_estimates,
            learned.base_estimates,
        )
        for warmup in (True, False):
            count = warmups if warmup else measured_repetitions
            for repetition in range(count):
                order = execution_order_for_repetition(configs, repetition)
                for order_index, config in enumerate(order, start=1):
                    treatment = plan_treatments[config]
                    rows.append(
                        run_once(
                            context,
                            output_dir,
                            database,
                            item,
                            config,
                            execution_queries[config],
                            treatment,
                            repetition=repetition,
                            warmup=warmup,
                            execution_order=order_index,
                            base_join_changed=base_changed,
                            full_plan_changed=full_changed,
                            treatment_category=treatment_category,
                            preflight_passed=bool(preflight["preflight_passed"]),
                            timeout_seconds=timeout_seconds,
                            cache_policy=cache_policy,
                        )
                    )
    return rows


def execution_order_for_repetition(
    configs: list[CardinalityConfig], repetition: int
) -> list[CardinalityConfig]:
    primary = [config for config in ("native", "learned_base") if config in configs]
    if repetition % 2 == 1:
        primary.reverse()
    return [*primary, *[config for config in configs if config not in primary]]


def run_once(
    context: Any,
    output_dir: Path,
    database: Any,
    item: Any,
    config: CardinalityConfig,
    query: Any,
    treatment: Any,
    *,
    repetition: int,
    warmup: bool,
    execution_order: int,
    base_join_changed: bool,
    full_plan_changed: bool,
    treatment_category: str,
    preflight_passed: bool,
    timeout_seconds: float,
    cache_policy: str,
) -> dict[str, Any]:
    if cache_policy == "cold-if-supported":
        database.reset_cache()
    started = time.perf_counter()
    document = None
    timeout = False
    error = None
    try:
        document = explain_json(
            database,
            query,
            analyze=True,
            buffers=True,
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        timeout = True
    except Exception as exc:
        error = str(exc)
    wall_ms = (time.perf_counter() - started) * 1000.0
    metrics = analyze_metrics(document) if document is not None else {}
    phase = "warmup" if warmup else "measured"
    analyze_path = output_dir / "plans" / item.label / config / f"{phase}_{repetition}.json"
    if document is not None:
        save_plan(analyze_path, document)
        analyze_path_value = context.relative(analyze_path)
    else:
        analyze_path_value = None
    status = "timeout" if timeout else ("error" if error else "ok")
    return {
        "query_id": item.label,
        "source_query_id": item.query_id,
        "query_template": item.template,
        "relation_count": item.query_size,
        "configuration": config,
        "repetition": repetition,
        "warmup": warmup,
        "execution_order": execution_order,
        "execution_sql": str(query),
        "hint": extract_hint(query),
        "base_join": treatment.first_base_join,
        "base_join_changed": base_join_changed,
        "plan_hash": treatment.plan_hash,
        "full_plan_changed": full_plan_changed,
        "treatment_category": treatment_category,
        "plan_path": treatment.plan_path,
        "analyze_plan_path": analyze_path_value,
        "wall_clock_duration_ms": wall_ms,
        "planning_time_ms": metrics.get("planning_time_ms"),
        "execution_time_ms": metrics.get("execution_time_ms"),
        "timeout": timeout,
        "timeout_seconds": timeout_seconds,
        "shared_hit_blocks": metrics.get("shared_hit_blocks"),
        "shared_read_blocks": metrics.get("shared_read_blocks"),
        "shared_dirtied_blocks": metrics.get("shared_dirtied_blocks"),
        "shared_written_blocks": metrics.get("shared_written_blocks"),
        "temp_read_blocks": metrics.get("temp_read_blocks"),
        "temp_written_blocks": metrics.get("temp_written_blocks"),
        "preflight_passed": preflight_passed,
        "treatment_valid": treatment.treatment_valid,
        "status": status,
        "error": error,
    }


def classify_treatment(
    base_join_changed: bool,
    full_plan_changed: bool,
    native_estimates: dict[str, float],
    learned_estimates: dict[str, float],
) -> str:
    if base_join_changed:
        return "base join changed"
    if full_plan_changed:
        return "plan changed but base join unchanged"
    meaningful = any(
        abs(math.log(max(float(learned_estimates[key]), 1.0) / max(float(native_estimates[key]), 1.0)))
        > math.log(1.01)
        for key in learned_estimates
        if key in native_estimates
    )
    return "estimates changed but plan unchanged" if meaningful else "no meaningful estimate difference"


def build_per_query(rows: list[dict[str, Any]], *, timeout_seconds: float) -> list[dict[str, Any]]:
    measured = [row for row in rows if not _as_bool(row["warmup"])]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        grouped[str(row["query_id"])].append(row)
    output: list[dict[str, Any]] = []
    for query_id, group in sorted(grouped.items()):
        first = group[0]
        by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_config[str(row["configuration"])].append(row)
        native_values = [_censored_execution(row, timeout_seconds) for row in by_config["native"]]
        learned_values = [_censored_execution(row, timeout_seconds) for row in by_config["learned_base"]]
        native_wall = [_censored_wall(row, timeout_seconds) for row in by_config["native"]]
        learned_wall = [_censored_wall(row, timeout_seconds) for row in by_config["learned_base"]]
        native_median = statistics.median(native_values)
        learned_median = statistics.median(learned_values)
        speedup = native_median / learned_median
        output.append(
            {
                "query_id": query_id,
                "source_query_id": first["source_query_id"],
                "query_template": first["query_template"],
                "relation_count": first["relation_count"],
                "native_base_join": by_config["native"][0]["base_join"],
                "learned_base_join": by_config["learned_base"][0]["base_join"],
                "base_join_changed": first["base_join_changed"],
                "native_plan_hash": by_config["native"][0]["plan_hash"],
                "learned_plan_hash": by_config["learned_base"][0]["plan_hash"],
                "full_plan_changed": first["full_plan_changed"],
                "treatment_category": first["treatment_category"],
                "native_median_execution_time_ms": native_median,
                "learned_median_execution_time_ms": learned_median,
                "native_median_wall_clock_ms": statistics.median(native_wall),
                "learned_median_wall_clock_ms": statistics.median(learned_wall),
                "speedup_native_over_learned": speedup,
                "runtime_classification": classify_speedup(speedup),
                "native_timeout_count": sum(_as_bool(row["timeout"]) for row in by_config["native"]),
                "learned_timeout_count": sum(_as_bool(row["timeout"]) for row in by_config["learned_base"]),
                "valid_measured_repetitions": sum(row["status"] == "ok" for row in group),
                "status": "complete" if native_values and learned_values else "invalid",
            }
        )
    return output


def build_summary(
    rows: list[dict[str, Any]],
    *,
    per_query_rows: list[dict[str, Any]] | None = None,
    timeout_seconds: float,
    samples: int = 10_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    per_query = per_query_rows if per_query_rows is not None else build_per_query(rows, timeout_seconds=timeout_seconds)
    valid = [row for row in per_query if row["status"] == "complete"]
    native = [float(row["native_median_execution_time_ms"]) for row in valid]
    learned = [float(row["learned_median_execution_time_ms"]) for row in valid]
    speedups = [float(row["speedup_native_over_learned"]) for row in valid]
    measured = [row for row in rows if not _as_bool(row["warmup"])]
    timeout_counts: dict[str, int] = defaultdict(int)
    for row in measured:
        timeout_counts[str(row["configuration"])] += int(_as_bool(row["timeout"]))
    exact_reference = _optional_exact_reference_summary(measured, timeout_seconds)
    return {
        "experiment": "runtime_impact",
        "primary_metric": "PostgreSQL Execution Time from one EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) execution",
        "secondary_metric": "wall-clock duration of the same execution",
        "speedup_definition": "native median execution time / learned-base median execution time",
        "practical_runtime_threshold": PRACTICAL_RUNTIME_THRESHOLD,
        "timeout_censoring_policy": f"timeouts are assigned {timeout_seconds * 1000.0} ms in the primary analysis",
        "query_count": len(valid),
        "measured_repetition_count": len(measured),
        "total_workload_execution_time_ms": {"native": sum(native), "learned_base": sum(learned)},
        "median_query_execution_time_ms": {
            "native": None if not native else statistics.median(native),
            "learned_base": None if not learned else statistics.median(learned),
        },
        "p90_query_execution_time_ms": {"native": percentile(native, 0.90), "learned_base": percentile(learned, 0.90)},
        "p95_query_execution_time_ms": {"native": percentile(native, 0.95), "learned_base": percentile(learned, 0.95)},
        "geometric_mean_speedup_native_over_learned": geometric_mean(speedups),
        "mean_log_runtime_ratio_bootstrap": paired_bootstrap_ci(
            [math.log(value) for value in native],
            [math.log(value) for value in learned],
            statistics.fmean,
            samples=samples,
            random_seed=random_seed,
        ),
        "timeout_counts": dict(sorted(timeout_counts.items())),
        "runtime_classifications": {
            name: sum(row["runtime_classification"] == name for row in valid)
            for name in ("improved", "degraded", "unchanged")
        },
        "subgroups": _subgroup_summaries(valid),
        "sensitivity_excluding_timeouts": _sensitivity_without_timeouts(valid),
        "exact_base_reference_optional": exact_reference,
        "attribution_rule": (
            "Runtime changes are attributed to learned estimates only for treatment-valid queries where the plan or "
            "base-join decision differs."
        ),
    }


def classify_speedup(speedup: float) -> str:
    if speedup > 1.0 / (1.0 - PRACTICAL_RUNTIME_THRESHOLD):
        return "improved"
    if speedup < 1.0 / (1.0 + PRACTICAL_RUNTIME_THRESHOLD):
        return "degraded"
    return "unchanged"


def _subgroup_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all_queries"].append(row)
        groups["changed_base_joins" if _as_bool(row["base_join_changed"]) else "unchanged_base_joins"].append(row)
        groups["changed_full_plans" if _as_bool(row["full_plan_changed"]) else "unchanged_full_plans"].append(row)
        groups[f"template={row['query_template']}"] .append(row)
    return {
        name: {
            "query_count": len(group),
            "geometric_mean_speedup_native_over_learned": geometric_mean(
                [float(row["speedup_native_over_learned"]) for row in group]
            ),
        }
        for name, group in sorted(groups.items())
    }


def _sensitivity_without_timeouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [
        row for row in rows if int(row["native_timeout_count"]) == 0 and int(row["learned_timeout_count"]) == 0
    ]
    return {
        "query_count": len(complete),
        "geometric_mean_speedup_native_over_learned": geometric_mean(
            [float(row["speedup_native_over_learned"]) for row in complete]
        ),
    }


def _optional_exact_reference_summary(
    measured: list[dict[str, Any]], timeout_seconds: float
) -> dict[str, Any] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in measured:
        if row["configuration"] == "exact_base":
            grouped[str(row["query_id"])].append(_censored_execution(row, timeout_seconds))
    if not grouped:
        return None
    medians = [statistics.median(values) for values in grouped.values()]
    return {
        "query_count": len(medians),
        "total_workload_execution_time_ms": sum(medians),
        "median_query_execution_time_ms": statistics.median(medians),
        "p90_query_execution_time_ms": percentile(medians, 0.90),
        "p95_query_execution_time_ms": percentile(medians, 0.95),
        "interpretation": "Reported separately and not included in native-versus-learned speedup.",
    }


def _censored_execution(row: dict[str, Any], timeout_seconds: float) -> float:
    if _as_bool(row["timeout"]):
        return timeout_seconds * 1000.0
    if row["status"] != "ok" or row.get("execution_time_ms") in (None, ""):
        raise ValueError(f"Missing execution time for non-timeout row {row['query_id']}")
    return float(row["execution_time_ms"])


def _censored_wall(row: dict[str, Any], timeout_seconds: float) -> float:
    return timeout_seconds * 1000.0 if _as_bool(row["timeout"]) else float(row["wall_clock_duration_ms"])


def _preflight_query(workload: list[Any]) -> Any:
    return select_preflight_query(workload)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


if __name__ == "__main__":
    main()

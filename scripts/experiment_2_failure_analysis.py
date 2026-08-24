from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import postbound as pb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cardinality import ExactCardinalityCache
from evaluation.stats_ceb.cli import (
    add_common_db_model_args,
    load_database_and_registry,
    prepare_run_context,
    select_workload,
)
from evaluation.stats_ceb.failure_analysis import (
    CATEGORY_FIELDS,
    SYSTEMATIC_MEDIAN_MULTIPLIER,
    SYSTEMATIC_MIN_GROUP_SIZE,
    build_category_metrics,
    worst_queries,
)
from evaluation.stats_ceb.metrics import q_error, summarize
from evaluation.stats_ceb.reports import write_csv, write_failures, write_json
from evaluation.stats_ceb.run_context import build_manifest, write_or_update_manifest
from evaluation.stats_ceb.single_table import SINGLE_TABLE_FIELDS, evaluate_single_table_workload
from evaluation.stats_ceb.workload import derive_base_queries, load_stats_ceb_workload
from integration.postbound.translator import translate_request
from registry.registry import ModelRegistry


SEED_SENSITIVITY_TARGET_COUNT = 10
SEED_SENSITIVITY_RUN_COUNT = 5
SEED_SENSITIVITY_FIELDS = [
    "rank",
    "occurrence_id",
    "normalized_sql_id",
    "query_template",
    "table",
    "sql",
    "exact_cardinality",
    "original_seed",
    "original_learned_estimate",
    "original_learned_q_error",
    "seed",
    "seed_offset",
    "learned_estimate",
    "learned_q_error",
    "estimate_to_exact_ratio",
    "estimate_change_from_original_percent",
    "inference_mode",
    "sample_count",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 2: learned-estimator failures and outliers")
    add_common_db_model_args(parser)
    parser.add_argument("--complete-workload-path", default=None)
    parser.add_argument("--single-table-workload-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--single-table-results-path",
        default=None,
        help="Experiment 1 single_table_results.csv; defaults to the current run's artifact.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 50:
        raise ValueError("--top-k must be at least 50 for the thesis failure artifact")
    context = prepare_run_context(args)
    output_dir = context.experiment_dir("experiment_2")
    if args.resume and (output_dir / "summary.json").exists():
        print(f"Experiment 2 already complete: {output_dir}")
        return
    database, registry = load_database_and_registry(args)
    source_path = Path(args.single_table_results_path).expanduser().resolve() if args.single_table_results_path else (
        context.run_dir / "experiment_1_accuracy" / "single_table_results.csv"
    )
    failures: list[dict[str, Any]] = []
    workload_sources: list[str | Path] = []
    if source_path.exists():
        rows = _load_existing_rows(source_path)
        workload_sources.append(source_path)
    else:
        if not args.complete_workload_path:
            raise FileNotFoundError(
                f"Experiment 1 results not found at {source_path}; provide --complete-workload-path to recompute"
            )
        workload = select_workload(
            load_stats_ceb_workload(
                args.complete_workload_path,
                workload_format="complete",
                template_map_path=args.template_map,
            ),
            args,
        )
        rows, failures = evaluate_single_table_workload(
            database,
            registry,
            derive_base_queries(workload),
            exact_cache=ExactCardinalityCache(database),
        )
        workload_sources.append(args.complete_workload_path)
    write_or_update_manifest(
        context,
        build_manifest(
            context,
            database=database,
            registry=registry,
            args=args,
            workload_sources=workload_sources,
            workload_query_count=len(rows),
        ),
    )
    categories = build_category_metrics(rows)
    worst = worst_queries(rows, limit=args.top_k)
    _write_diagnostics(context, output_dir, worst)

    seed_sensitivity = _run_sampling_seed_sensitivity(
        registry,
        rows,
        base_seed=args.random_seed,
        sample_count=args.sample_count,
        target_count=SEED_SENSITIVITY_TARGET_COUNT,
        seed_count=SEED_SENSITIVITY_RUN_COUNT,
    )
    seed_sensitivity_path = output_dir / "sampling_seed_sensitivity.csv"
    write_csv(seed_sensitivity_path, seed_sensitivity, SEED_SENSITIVITY_FIELDS)

    valid = [row for row in rows if row.get("status", "ok") == "ok"]
    systematic = [row for row in categories if _as_bool(row["systematic_failure_candidate"])]
    summary = {
        "experiment": "failure_analysis",
        "row_count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "overall_learned_q_error": summarize([float(row["learned_q_error"]) for row in valid]),
        "overall_native_q_error": summarize([float(row["native_q_error"]) for row in valid]),
        "systematic_failure_definition": {
            "minimum_group_size": SYSTEMATIC_MIN_GROUP_SIZE,
            "median_q_error_multiplier_overall": SYSTEMATIC_MEDIAN_MULTIPLIER,
            "status": "candidate classification requiring manual review",
        },
        "systematic_failure_candidates": systematic,
        "worst_query_count": len(worst),
        "sampling_seed_sensitivity": {
            "selection": "worst distinct filtered progressive-sampling occurrences by learned Q-error",
            "target_count": len({row["normalized_sql_id"] for row in seed_sensitivity}),
            "seed_count": SEED_SENSITIVITY_RUN_COUNT,
            "seeds": sorted({int(row["seed"]) for row in seed_sensitivity}),
            "sample_count_override": args.sample_count,
            "output_file": context.relative(seed_sensitivity_path),
        },
    }
    write_csv(output_dir / "grouped_failures.csv", categories, CATEGORY_FIELDS)
    worst_fields = [*SINGLE_TABLE_FIELDS, "learned_native_q_error_ratio", "worst_learned_q_error", "worst_learned_native_disadvantage", "candidate_classification", "classification_status", "classification_evidence", "diagnostic_path"]
    write_csv(output_dir / "worst_queries.csv", worst, worst_fields)
    write_json(output_dir / "summary.json", summary)
    write_failures(output_dir, failures)


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_sampling_seed_sensitivity(
    registry: ModelRegistry,
    rows: list[dict[str, Any]],
    *,
    base_seed: int,
    sample_count: int | None,
    target_count: int,
    seed_count: int,
) -> list[dict[str, Any]]:
    targets = _seed_sensitivity_targets(rows, limit=target_count)
    seeds = [int(base_seed) + offset for offset in range(seed_count)]
    output: list[dict[str, Any]] = []

    try:
        for seed_offset, seed in enumerate(seeds):
            registry.configure_inference(random_seed=seed, sample_count=sample_count)
            for rank, row in enumerate(targets, start=1):
                query = pb.parse_query(str(row["sql"]))
                tables = list(query.tables())
                if len(tables) != 1:
                    raise ValueError(
                        f"Seed-sensitivity target {row['occurrence_id']} is not a single-table query"
                    )
                translation = translate_request(query, tables[0])
                if not translation.can_estimate:
                    raise ValueError(
                        f"Seed-sensitivity target {row['occurrence_id']} cannot be translated: "
                        f"{translation.fallback_reason}"
                    )

                result = registry.estimate(
                    translation.table_name or str(row["table"]),
                    translation.predicates,
                )
                if not result.used_model:
                    raise ValueError(
                        f"Seed-sensitivity target {row['occurrence_id']} unexpectedly used fallback: "
                        f"{result.reason}"
                    )

                exact = float(row["exact_cardinality"])
                original_estimate = float(row["learned_estimate"])
                estimate = float(result.cardinality)
                diagnostics = dict(result.diagnostics)
                output.append(
                    {
                        "rank": rank,
                        "occurrence_id": row["occurrence_id"],
                        "normalized_sql_id": row["normalized_sql_id"],
                        "query_template": row["query_template"],
                        "table": row["table"],
                        "sql": row["sql"],
                        "exact_cardinality": exact,
                        "original_seed": row.get("estimator_seed"),
                        "original_learned_estimate": original_estimate,
                        "original_learned_q_error": float(row["learned_q_error"]),
                        "seed": seed,
                        "seed_offset": seed_offset,
                        "learned_estimate": estimate,
                        "learned_q_error": q_error(estimate, exact),
                        "estimate_to_exact_ratio": None if exact == 0 else estimate / exact,
                        "estimate_change_from_original_percent": None
                        if original_estimate == 0
                        else 100.0 * (estimate - original_estimate) / original_estimate,
                        "inference_mode": diagnostics.get("inference_mode"),
                        "sample_count": diagnostics.get("sample_count"),
                    }
                )
    finally:
        registry.configure_inference(random_seed=base_seed, sample_count=sample_count)

    return output


def _seed_sensitivity_targets(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("status", "ok") == "ok"
        and _as_bool(row.get("is_filtered"))
        and _row_used_model(row)
        and str(row.get("inference_mode")) == "progressive_sampling"
    ]
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["learned_q_error"]), str(row["occurrence_id"])),
    )
    selected: list[dict[str, Any]] = []
    seen_normalized_sql: set[str] = set()
    for row in ordered:
        normalized_sql_id = str(row["normalized_sql_id"])
        if normalized_sql_id in seen_normalized_sql:
            continue
        seen_normalized_sql.add(normalized_sql_id)
        selected.append(row)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(
            f"Only {len(selected)} distinct filtered progressive-sampling occurrences are available; "
            f"need {limit} for seed sensitivity"
        )
    return selected


def _row_used_model(row: dict[str, Any]) -> bool:
    if "used_model" in row:
        return _as_bool(row.get("used_model"))
    return bool(str(row.get("model_used") or "").strip())


def _write_diagnostics(context: Any, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        occurrence = str(row["occurrence_id"])
        path = diagnostics_dir / f"{occurrence}.json"
        diagnostics = json.loads(row.get("diagnostics_json") or "{}")
        write_json(
            path,
            {
                "occurrence_id": occurrence,
                "sql": row["sql"],
                "exact_cardinality": row["exact_cardinality"],
                "native_estimate": row["native_estimate"],
                "learned_estimate": row["learned_estimate"],
                "learned_raw_estimate": row["learned_raw_estimate"],
                "diagnostics": diagnostics,
                "candidate_classification": row["candidate_classification"],
                "classification_evidence": json.loads(row["classification_evidence"]),
            },
        )
        row["diagnostic_path"] = context.relative(path)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


if __name__ == "__main__":
    main()

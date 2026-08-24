from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.cli import add_common_db_model_args, load_database_and_registry, prepare_run_context, select_workload
from evaluation.stats_ceb.preflight import ensure_preflight, select_preflight_query
from evaluation.stats_ceb.run_context import build_manifest, write_or_update_manifest
from evaluation.stats_ceb.workload import load_stats_ceb_workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PostgreSQL, pg_lab, models, and optimizer treatments")
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
    sample = select_preflight_query(workload)
    expected = {"posts:p": 879.0} if sample.label == "stats_ceb_00135" else None
    payload = ensure_preflight(
        context,
        database=database,
        registry=registry,
        sample_query=sample,
        injection_validation=args.injection_validation,
        timeout_seconds=args.timeout_seconds,
        expected_exact=expected,
    )
    print(f"Preflight passed: {context.run_dir / 'preflight' / 'preflight.json'}")
    print(f"Validation query: {payload['project_generated_treatment']['query_id']}")
if __name__ == "__main__":
    main()

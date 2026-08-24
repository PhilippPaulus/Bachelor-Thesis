from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from backends.postgres import connect_postgres_database
from evaluation.stats_ceb.run_context import RunContext
from evaluation.stats_ceb.workload import StatsCebQuery
from registry.registry import ModelRegistry


def add_common_db_model_args(parser: argparse.ArgumentParser) -> None:
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument("--connection-file")
    connection.add_argument("--conn-string", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-root", default="artifacts/evaluations")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--template-map", default=None)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--timeout-seconds", "--timeout", type=float, default=300.0)
    parser.add_argument("--warmups", "--warm-ups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--cache-policy",
        choices=("controlled-warm", "cold-if-supported"),
        default="controlled-warm",
    )
    parser.add_argument(
        "--injection-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")


def load_database_and_registry(args: argparse.Namespace) -> tuple[Any, ModelRegistry]:
    conn_string = _connection_string(args)
    database = connect_postgres_database(conn_string, cache_enabled=False)
    registry = ModelRegistry.load(args.model_dir)
    if not registry.available_tables():
        raise ValueError(f"No models found in model directory: {args.model_dir}")
    registry.configure_inference(random_seed=args.random_seed, sample_count=args.sample_count)
    return database, registry


def prepare_run_context(args: argparse.Namespace) -> RunContext:
    return RunContext.create(
        args.output_root,
        args.run_id,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )


def select_workload(workload: list[StatsCebQuery], args: argparse.Namespace) -> list[StatsCebQuery]:
    requested = {
        token.strip()
        for value in args.query_id
        for token in str(value).split(",")
        if token.strip()
    }
    selected = [
        item
        for item in workload
        if not requested
        or item.label in requested
        or item.query_id in requested
        or str(item.line_number) in requested
    ]
    if requested and not selected:
        raise ValueError(f"No workload queries matched --query-id values: {sorted(requested)}")
    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise ValueError("--query-limit must be positive")
        selected = selected[: args.query_limit]
    return selected


def _connection_string(args: argparse.Namespace) -> str:
    if getattr(args, "connection_file", None):
        path = Path(args.connection_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Connection file does not exist: {path}")
        value = path.read_text(encoding="utf-8").strip()
    else:
        value = str(args.conn_string).strip()
    if not value:
        raise ValueError("Database connection string is empty")
    return value

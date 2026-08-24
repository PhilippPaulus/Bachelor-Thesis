from __future__ import annotations

import argparse
from pathlib import Path

from backends.postgres import connect_postgres_database
from core.config import NaruConfig
from .training_pipeline import train_all_tables, train_single_table


def build_parser() -> argparse.ArgumentParser:
    defaults = NaruConfig()
    parser = argparse.ArgumentParser(description="Train Naru-style table models for PostgreSQL tables")
    parser.add_argument("--conn-string", required=True)
    parser.add_argument("--schema", default=defaults.schema_name)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--table")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--sample-count", type=int, default=defaults.sample_count)
    parser.add_argument("--max-rows", type=int, default=defaults.max_rows)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--numeric-max-unique-for-dictionary",
        type=int,
        default=defaults.numeric_max_unique_for_dictionary,
    )
    parser.add_argument("--numeric-bin-count", type=int, default=defaults.numeric_bin_count)
    parser.add_argument("--min-epochs-before-timeout", type=int, default=defaults.min_epochs_before_timeout)
    parser.add_argument(
        "--epoch-timeout-seconds-after-min-epochs",
        type=float,
        default=defaults.epoch_timeout_seconds_after_min_epochs,
        help="Stop after an epoch exceeds this duration once the minimum epoch count has completed.",
    )
    parser.add_argument(
        "--hidden-dims",
        default=",".join(str(dim) for dim in defaults.hidden_dims),
        help="Comma-separated hidden layer sizes, e.g. '128,128'",
    )
    return parser


def _parse_hidden_dims(raw_dims: str) -> tuple[int, ...]:
    return tuple(int(dim.strip()) for dim in raw_dims.split(",") if dim.strip())


def main() -> None:
    args = build_parser().parse_args()
    config = NaruConfig(
        schema_name=args.schema,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        sample_count=args.sample_count,
        max_rows=args.max_rows,
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        random_seed=args.random_seed,
        numeric_max_unique_for_dictionary=args.numeric_max_unique_for_dictionary,
        numeric_bin_count=args.numeric_bin_count,
        min_epochs_before_timeout=args.min_epochs_before_timeout,
        epoch_timeout_seconds_after_min_epochs=args.epoch_timeout_seconds_after_min_epochs,
    )
    database = connect_postgres_database(args.conn_string)
    output_dir = Path(args.output_dir)
    if args.table:
        train_single_table(
            database,
            args.table,
            output_dir,
            schema_name=args.schema,
            config=config,
        )
    else:
        train_all_tables(
            database,
            args.schema,
            output_dir,
            config=config,
        )


if __name__ == "__main__":
    main()

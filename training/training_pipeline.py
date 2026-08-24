from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from postbound import ColumnReference, TableReference
from postbound.db import Database

from core.config import NaruConfig
from core.domain import ColumnInfo, TableInfo, classify_backend_type
from core.logging_utils import configure_logging, get_logger, log_kv
from integration.postbound.qal_utils import build_table_query
from model.encoding.encoders import TableEncoder
from model.naru.persistence import artifact_paths, save_artifacts
from model.naru.training import train_model

logger = get_logger(__name__)


def _schema_tables(database: Database, schema_name: str) -> list[TableReference]:
    schema = database.schema()
    try:
        tables = schema.tables(schema=schema_name)
    except TypeError:
        tables = {
            table
            for table in schema.tables()
            if not getattr(table, "virtual", False)
            and (not getattr(table, "schema", "") or getattr(table, "schema", "") == schema_name)
        }
    return sorted(tables, key=lambda table: (getattr(table, "schema", ""), table.full_name))


def _resolve_table(database: Database, schema_name: str, table_name: str) -> TableReference:
    qualified_name = f"{schema_name}.{table_name}"
    for table in _schema_tables(database, schema_name):
        if table.full_name == table_name:
            return table
    raise KeyError(f"Table '{qualified_name}' is not visible through the PostBOUND schema interface")


def _extract_row_count(result: Any) -> int:
    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("COUNT query returned no rows")
        first_row = result[0]
        if isinstance(first_row, (list, tuple)):
            return int(first_row[0])
        return int(first_row)
    return int(result)


def _exact_row_count(database: Database, table: TableReference) -> int:
    query = build_table_query(table, count_star=True)
    return _extract_row_count(database.execute_query(query))


def _table_info(
    database: Database,
    table: TableReference,
    *,
    schema_name: str,
    include_exact_row_count: bool,
) -> TableInfo:
    schema = database.schema()
    resolved_schema = table.schema or schema_name
    columns: list[ColumnInfo] = []
    for ordinal_position, column in enumerate(schema.columns(table), start=1):
        backend_type = str(schema.datatype(column))
        supported, kind = classify_backend_type(backend_type)
        columns.append(
            ColumnInfo(
                schema_name=resolved_schema,
                table_name=table.full_name,
                column_name=column.name,
                backend_type=backend_type,
                column_kind=kind,
                ordinal_position=ordinal_position,
                is_nullable=bool(schema.is_nullable(column)),
                supported=supported,
            )
        )
    estimated_row_count = database.statistics().total_rows(table)
    exact_row_count = _exact_row_count(database, table) if include_exact_row_count else None
    return TableInfo(
        schema_name=resolved_schema,
        table_name=table.full_name,
        estimated_row_count=None if estimated_row_count is None else int(estimated_row_count),
        exact_row_count=exact_row_count,
        columns=columns,
    )


def _fetch_training_rows(
    database: Database,
    table: TableReference,
    columns: list[ColumnInfo],
    *,
    sampling_strategy: str,
    max_rows: int | None,
) -> pd.DataFrame:
    if sampling_strategy != "limit":
        raise ValueError(
            f"Unsupported sampling strategy '{sampling_strategy}'. "
            "Only 'limit' is currently implemented through the PostBOUND training pipeline."
        )
    query_columns = [ColumnReference(column.column_name, table) for column in columns]
    query = build_table_query(table, columns=query_columns, limit=max_rows)
    rows = database.execute_query(query)
    if not isinstance(rows, (list, tuple)):
        raise TypeError("Training row fetch returned an unsupported result shape")
    return pd.DataFrame(list(rows), columns=[column.column_name for column in columns])


def _should_skip_table(table_name: str, config: NaruConfig) -> bool:
    if config.table_allowlist and table_name not in config.table_allowlist:
        return True
    if config.table_denylist and table_name in config.table_denylist:
        return True
    return False


def train_single_table(
    database: Database,
    table_name: str,
    output_dir: str | Path,
    *,
    schema_name: str = "public",
    config: NaruConfig | None = None,
) -> Path:
    config = config or NaruConfig(schema_name=schema_name)
    configure_logging(config.log_level)
    table = _resolve_table(database, schema_name, table_name)
    table_info = _table_info(
        database,
        table,
        schema_name=schema_name,
        include_exact_row_count=not config.use_rowcount_estimate,
    )
    dataframe = _fetch_training_rows(
        database,
        table,
        table_info.supported_columns,
        sampling_strategy=config.sampling_strategy,
        max_rows=config.max_rows,
    )

    encoder, encoded_rows = TableEncoder.fit(table_info, dataframe, config)
    progress_label = f"{schema_name}.{table_name}"
    print(f"\nTraining {progress_label} ({len(dataframe)} rows, {len(table_info.supported_columns)} columns)", flush=True)
    model, summary = train_model(
        encoded_rows,
        encoder.domain_sizes,
        config,
        progress_label=progress_label,
    )

    artifacts = artifact_paths(Path(output_dir), schema_name, table_name)
    save_artifacts(artifacts, model, encoder, config, summary if config.save_training_summary else None)
    log_kv(logger, "Finished table training", table=table_name, rows=encoder.row_count)
    return artifacts.table_dir


def train_all_tables(
    database: Database,
    schema_name: str,
    output_dir: str | Path,
    *,
    config: NaruConfig | None = None,
) -> list[Path]:
    config = config or NaruConfig(schema_name=schema_name)
    configure_logging(config.log_level)
    tables = [table.full_name for table in _schema_tables(database, schema_name)]

    trained_paths: list[Path] = []
    trainable_tables = [table_name for table_name in tables if not _should_skip_table(table_name, config)]
    for table_index, table_name in enumerate(trainable_tables, start=1):
        if _should_skip_table(table_name, config):
            continue
        print(f"\n=== Table {table_index}/{len(trainable_tables)}: {schema_name}.{table_name} ===", flush=True)
        trained_paths.append(
            train_single_table(
                database,
                table_name,
                output_dir,
                schema_name=schema_name,
                config=config,
            )
        )
    return trained_paths

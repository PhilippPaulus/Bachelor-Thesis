from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import postbound as pb

from evaluation.stats_ceb.cardinality import (
    BaseEstimateRecord,
    ExactCardinalityCache,
    base_estimates,
    cardinality_target_query,
    optimized_query_for_config,
    table_key,
)
from evaluation.stats_ceb.plans import (
    explain_json,
    extract_hint,
    first_base_join_key,
    root_plan_rows,
    save_plan,
    save_sql,
    stable_plan_hash,
)
from evaluation.stats_ceb.reports import write_json
from evaluation.stats_ceb.run_context import RunContext
from evaluation.stats_ceb.workload import StatsCebQuery
from integration.postbound.qal_utils import build_table_query
from integration.postbound.translator import base_filter_predicate


REQUIRED_STATS_TABLES = {
    "badges",
    "comments",
    "posthistory",
    "postlinks",
    "posts",
    "tags",
    "users",
    "votes",
}


class PreflightError(RuntimeError):
    pass


def select_preflight_query(workload: list[StatsCebQuery]) -> StatsCebQuery:
    """Selects the benchmark regression query when available.

    ``stats_ceb_00135`` is the documented end-to-end acceptance query and is
    known to contain filtered relations whose native and exact estimates
    differ.  Small/custom workloads retain the previous filtered-query
    fallback.
    """
    regression = next((item for item in workload if item.label == "stats_ceb_00135"), None)
    if regression is not None:
        return regression
    for item in workload:
        target = cardinality_target_query(item.query)
        if any(base_filter_predicate(target, table) is not None for table in target.tables()):
            return item
    raise ValueError("Preflight requires at least one query with a filtered base relation")


def ensure_preflight(
    context: RunContext,
    *,
    database: Any,
    registry: Any,
    sample_query: StatsCebQuery,
    injection_validation: bool,
    timeout_seconds: float,
    expected_exact: dict[str, float] | None = None,
) -> dict[str, Any]:
    path = context.run_dir / "preflight" / "preflight.json"
    if context.resume and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("preflight_passed") is True:
            return payload
        raise PreflightError(f"Existing preflight did not pass: {path}")
    return run_preflight(
        context,
        database=database,
        registry=registry,
        sample_query=sample_query,
        injection_validation=injection_validation,
        timeout_seconds=timeout_seconds,
        expected_exact=expected_exact,
    )


def run_preflight(
    context: RunContext,
    *,
    database: Any,
    registry: Any,
    sample_query: StatsCebQuery,
    injection_validation: bool = True,
    timeout_seconds: float = 300.0,
    expected_exact: dict[str, float] | None = None,
) -> dict[str, Any]:
    output_path = context.run_dir / "preflight" / "preflight.json"
    payload: dict[str, Any] = {
        "preflight_passed": False,
        "hint_syntax_valid": False,
        "hint_roundtrip_valid": False,
        "treatment_valid": False,
        "injection_validation_enabled": injection_validation,
        "errors": [],
    }
    try:
        payload["database"] = _database_validation(database)
        payload["models"] = _model_validation(registry)
        payload["direct_pg_lab_roundtrip"] = _direct_roundtrip(database, timeout_seconds)
        payload["hint_syntax_valid"] = True
        payload["hint_roundtrip_valid"] = True
        project = _project_treatment_validation(
            context,
            database,
            registry,
            sample_query,
            injection_validation=injection_validation,
            timeout_seconds=timeout_seconds,
            expected_exact=expected_exact or {},
        )
        payload["project_generated_treatment"] = project
        payload["treatment_valid"] = bool(project["treatment_valid"])
        payload["preflight_passed"] = all(
            bool(payload[key])
            for key in ("hint_syntax_valid", "hint_roundtrip_valid", "treatment_valid")
        )
    except Exception as exc:
        payload["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        write_json(output_path, payload)
        raise PreflightError(f"Evaluation preflight failed: {exc}") from exc
    write_json(output_path, payload)
    if not payload["preflight_passed"]:
        raise PreflightError(f"Evaluation preflight did not pass: {output_path}")
    return payload


def validate_records_roundtrip(
    database: Any,
    query: Any,
    records: list[BaseEstimateRecord],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    tables_by_key = {table_key(table): table for table in query.tables()}
    results: list[dict[str, Any]] = []
    for record in records:
        table = tables_by_key.get(record.table)
        if table is None:
            raise ValueError(f"Treatment record references unknown table: {record.table}")
        predicate = base_filter_predicate(query, table)
        standalone = build_table_query(table, predicate=predicate)
        params = pb.PlanParameterization()
        params.add_cardinality([table], record.injected_cardinality)
        hinted = database.hinting().generate_hints(standalone, plan_parameters=params)
        sql = str(hinted)
        alias = str(getattr(table, "alias", "") or table.full_name)
        syntax_valid = "/*=pg_lab=" in sql and f"Card({alias} #" in sql
        if not syntax_valid:
            raise ValueError(f"Invalid standalone pg_lab syntax for {record.table}: {sql}")
        document = explain_json(database, hinted, timeout_seconds=timeout_seconds)
        observed = root_plan_rows(document)
        tolerance = max(1.0, abs(record.injected_cardinality) * 1e-6)
        valid = abs(observed - record.injected_cardinality) <= tolerance
        results.append(
            {
                "table": record.table,
                "raw_cardinality": record.cardinality,
                "injected_cardinality": record.injected_cardinality,
                "observed_plan_rows": observed,
                "rounding_tolerance": tolerance,
                "hint_syntax_valid": syntax_valid,
                "hint_roundtrip_valid": valid,
                "sql": sql,
            }
        )
        if not valid:
            raise ValueError(
                f"Standalone pg_lab round trip failed for {record.table}: "
                f"supplied {record.injected_cardinality}, observed {observed}"
            )
    return results


def _database_validation(database: Any) -> dict[str, Any]:
    available = {str(table.full_name).lower() for table in database.schema().tables()}
    missing = sorted(REQUIRED_STATS_TABLES - available)
    if missing:
        raise ValueError(f"Database is missing required STATS tables: {', '.join(missing)}")
    counts: dict[str, int] = {}
    for table in sorted(REQUIRED_STATS_TABLES):
        value = float(database.execute_query(f'SELECT COUNT(*) FROM "{table}"', cache_enabled=False))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid exact row count for {table}: {value!r}")
        counts[table] = int(value)
    return {
        "reachable": True,
        "database_name": database.database_name(),
        "postgresql_version": str(database.database_system_version()),
        "required_tables_present": True,
        "exact_table_row_counts": counts,
    }


def _model_validation(registry: Any) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    for qualified_name in registry.available_tables():
        estimator = registry.get_model(qualified_name)
        row_count = int(estimator.encoder.row_count)
        if row_count <= 0:
            raise ValueError(f"Model {qualified_name} has invalid encoder row count {row_count}")
        tables.append(
            {
                "table": qualified_name,
                "loaded": True,
                "row_count": row_count,
                "device": str(estimator.device),
                "sample_count": estimator.sample_count,
                "estimation_seed": estimator.estimation_seed,
            }
        )
    if not tables:
        raise ValueError("No model artifacts could be loaded")
    return {"all_models_loaded": True, "model_count": len(tables), "tables": tables}


def _direct_roundtrip(database: Any, timeout_seconds: float) -> dict[str, Any]:
    native = pb.parse_query("SELECT * FROM posts AS p WHERE p.score > 10")
    hinted = pb.parse_query("/*=pg_lab= Card(p #42) */ SELECT * FROM posts AS p WHERE p.score > 10")
    native_document = explain_json(database, native, timeout_seconds=timeout_seconds)
    hinted_document = explain_json(database, hinted, timeout_seconds=timeout_seconds)
    native_rows = root_plan_rows(native_document)
    hinted_rows = root_plan_rows(hinted_document)
    if abs(native_rows - 42.0) <= 0.5:
        raise ValueError("Direct pg_lab control is invalid because the native estimate is already 42")
    if abs(hinted_rows - 42.0) > 0.5:
        raise ValueError(f"Direct pg_lab hint was ignored: expected 42, observed {hinted_rows}")
    return {
        "native_sql": str(native),
        "hinted_sql": str(hinted),
        "native_plan_rows": native_rows,
        "hinted_plan_rows": hinted_rows,
        "hint_syntax_valid": "/*=pg_lab=" in str(hinted) and "Card(" in str(hinted),
        "hint_roundtrip_valid": True,
    }


def _project_treatment_validation(
    context: RunContext,
    database: Any,
    registry: Any,
    item: StatsCebQuery,
    *,
    injection_validation: bool,
    timeout_seconds: float,
    expected_exact: dict[str, float],
) -> dict[str, Any]:
    target = cardinality_target_query(item.query)
    filtered_tables = [table for table in target.tables() if base_filter_predicate(target, table) is not None]
    if not filtered_tables:
        raise ValueError(f"Preflight query {item.label} has no filtered relations")
    cache = ExactCardinalityCache(database)
    configurations: dict[str, Any] = {}
    record_sets: dict[str, list[BaseEstimateRecord]] = {}
    for config in ("native", "learned_base", "exact_base"):
        optimized, records = optimized_query_for_config(
            database,
            target,
            config,
            registry=registry,
            exact_cache=cache,
        )
        sql = str(optimized)
        if config == "native" and extract_hint(sql):
            raise ValueError("Native preflight query unexpectedly contains a pg_lab hint")
        if config != "native":
            _validate_generated_sql(target, sql, records)
        document = explain_json(database, optimized, timeout_seconds=timeout_seconds)
        plan_path = context.run_dir / "preflight" / f"{item.label}_{config}.json"
        sql_path = context.run_dir / "preflight" / f"{item.label}_{config}.sql"
        save_plan(plan_path, document)
        save_sql(sql_path, optimized)
        configurations[config] = {
            "sql": sql,
            "hint": extract_hint(sql),
            "plan_path": context.relative(plan_path),
            "sql_path": context.relative(sql_path),
            "plan_hash": stable_plan_hash(document),
            "root_plan_rows": root_plan_rows(document),
            "first_base_join": first_base_join_key(document),
            "base_estimates": {record.table: record.cardinality for record in records},
            "injected_base_estimates": {
                record.table: record.injected_cardinality for record in records
            },
        }
        record_sets[config] = records
    native_estimates = base_estimates(database, target, "native")
    configurations["native"]["base_estimates"] = native_estimates
    exact_estimates = {record.table: record.cardinality for record in record_sets["exact_base"]}
    for key, expected in expected_exact.items():
        observed = exact_estimates.get(key)
        if observed is None or observed != expected:
            raise ValueError(f"Expected exact {key} cardinality {expected}, observed {observed}")
    differing_exact = {
        key: {"native": native_estimates[key], "exact": exact_estimates[key]}
        for key in exact_estimates
        if key in native_estimates and abs(exact_estimates[key] - native_estimates[key]) > 0.5
    }
    if not differing_exact:
        raise ValueError("Project preflight query does not exercise a differing exact base estimate")
    roundtrips: dict[str, Any] = {}
    for config in ("learned_base", "exact_base"):
        records = record_sets[config]
        if not injection_validation:
            if config == "exact_base":
                records = [next(record for record in records if record.table in differing_exact)]
            else:
                filtered_keys = {table_key(table) for table in filtered_tables}
                records = [next(record for record in records if record.table in filtered_keys)]
        roundtrips[config] = validate_records_roundtrip(
            database,
            target,
            records,
            timeout_seconds=timeout_seconds,
        )
    return {
        "query_id": item.label,
        "original_count_sql": item.sql,
        "aggregate_free_sql": str(target),
        "aggregate_removed": "count(" not in str(target).lower(),
        "configurations": configurations,
        "standalone_relation_roundtrips": roundtrips,
        "relations_where_exact_differs_from_native": differing_exact,
        "treatment_valid": True,
    }


def _validate_generated_sql(query: Any, sql: str, records: list[BaseEstimateRecord]) -> None:
    if "/*=pg_lab=" not in sql or "Card(" not in sql:
        raise ValueError(f"Project-generated treatment is missing pg_lab Card hints: {sql}")
    aliases = {
        table_key(table): str(getattr(table, "alias", "") or table.full_name)
        for table in query.tables()
    }
    for record in records:
        if not math.isfinite(record.cardinality) or record.cardinality < 0:
            raise ValueError(f"Non-finite treatment estimate for {record.table}: {record.cardinality!r}")
        if not math.isfinite(record.injected_cardinality) or record.injected_cardinality <= 0:
            raise ValueError(f"Non-positive injected estimate for {record.table}: {record.injected_cardinality!r}")
        alias = aliases[record.table]
        if f"Card({alias} #" not in sql:
            raise ValueError(f"Generated hint does not contain alias {alias}: {sql}")

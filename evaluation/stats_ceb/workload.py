from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import postbound as pb
from postbound import qal as pb_qal

from evaluation.stats_ceb.cardinality import table_key
from integration.postbound.qal_utils import build_table_query
from integration.postbound.translator import base_filter_predicate

WorkloadFormat = Literal["auto", "single_table", "subquery", "complete"]


@dataclass(frozen=True, slots=True)
class StatsCebQuery:
    label: str
    sql: str
    query: pb_qal.SqlQuery
    query_id: str
    line_number: int
    query_size: int
    template: str
    actual_cardinality: float | None = None

    @property
    def normalized_sql(self) -> str:
        return normalize_sql(self.query)

    @property
    def normalized_sql_id(self) -> str:
        return normalized_sql_id(self.query)


@dataclass(frozen=True, slots=True)
class StatsCebBaseQuery:
    occurrence_id: str
    normalized_sql_id: str
    normalized_sql: str
    original_query_label: str
    original_query_id: str
    original_sql: str
    template: str
    table_name: str
    table_alias: str
    table_key: str
    predicate_text: str
    query: pb_qal.SqlQuery
    sql: str
    is_filtered: bool


def load_template_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping_path = Path(path).expanduser().resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Template map does not exist: {mapping_path}")
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        result: dict[str, str] = {}
        for row in reader:
            template = row.get("template")
            if not template:
                continue
            for key_name in ("label", "query_id"):
                key = row.get(key_name)
                if key:
                    result[str(key)] = template
        return result


def load_stats_ceb_workload(
    path: str | Path,
    *,
    workload_format: WorkloadFormat = "auto",
    label_prefix: str | None = None,
    template_map_path: str | Path | None = None,
    template: str | None = None,
) -> list[StatsCebQuery]:
    workload_path = Path(path).expanduser().resolve()
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload path does not exist: {workload_path}")
    if not workload_path.is_file():
        raise ValueError(f"Expected a workload file, got directory: {workload_path}")

    template_map = load_template_map(template_map_path)
    prefix = _normalize_label(label_prefix or workload_path.stem)
    default_template = template or prefix
    rows: list[StatsCebQuery] = []
    for line_number, raw_line in enumerate(workload_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        sql, query_id, actual = _parse_line(line, workload_format)
        query_id = query_id or str(line_number)
        label = f"{prefix}_{line_number:05d}"
        query = pb.parse_query(sql)
        resolved_template = template_map.get(label) or template_map.get(query_id) or default_template
        rows.append(
            StatsCebQuery(
                label=label,
                sql=sql,
                query=query,
                query_id=query_id,
                line_number=line_number,
                query_size=len(list(query.tables())),
                template=resolved_template,
                actual_cardinality=actual,
            )
        )
    if not rows:
        raise ValueError(f"No queries found in workload path: {workload_path}")
    return rows


def derive_base_queries(workload: list[StatsCebQuery]) -> list[StatsCebBaseQuery]:
    derived: list[StatsCebBaseQuery] = []
    for item in workload:
        for table in sorted(item.query.tables(), key=table_key):
            predicate = base_filter_predicate(item.query, table)
            query = build_table_query(table, predicate=predicate)
            normalized = normalize_sql(query)
            relation = table.qualified_name() if table.schema else table.full_name
            alias = str(getattr(table, "alias", "") or "")
            relation_key = table_key(table)
            derived.append(
                StatsCebBaseQuery(
                    occurrence_id=f"{item.label}__{relation_key.replace(':', '_')}",
                    normalized_sql_id=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    normalized_sql=normalized,
                    original_query_label=item.label,
                    original_query_id=item.query_id,
                    original_sql=item.sql,
                    template=item.template,
                    table_name=relation,
                    table_alias=alias,
                    table_key=relation_key,
                    predicate_text="" if predicate is None else str(predicate),
                    query=query,
                    sql=str(query),
                    is_filtered=predicate is not None,
                )
            )
    return derived


def normalize_sql(query_or_sql: pb_qal.SqlQuery | str) -> str:
    if isinstance(query_or_sql, str):
        query = pb.parse_query(query_or_sql)
    else:
        query = query_or_sql
    return " ".join(str(query).strip().rstrip(";").lower().split())


def normalized_sql_id(query_or_sql: pb_qal.SqlQuery | str) -> str:
    return hashlib.sha256(normalize_sql(query_or_sql).encode("utf-8")).hexdigest()


def join_count(query: pb_qal.SqlQuery) -> int:
    predicates = query.predicates()
    if predicates is None:
        return 0
    try:
        return len(list(predicates.joins()))
    except (AttributeError, TypeError):
        return max(len(list(query.tables())) - 1, 0)


def _parse_line(line: str, workload_format: WorkloadFormat) -> tuple[str, str, float | None]:
    parts = [part.strip() for part in line.split("||")]
    if workload_format == "auto":
        workload_format = _detect_format(parts)

    if workload_format == "single_table":
        if len(parts) < 3:
            raise ValueError(f"Expected SQL||query_id||actual line, got: {line!r}")
        return parts[0], parts[1], float(parts[2])

    if workload_format == "subquery":
        if len(parts) < 2:
            raise ValueError(f"Expected SQL||query_id line, got: {line!r}")
        return parts[0], parts[1], None

    if workload_format == "complete":
        if len(parts) < 2:
            raise ValueError(f"Expected actual||SQL line, got: {line!r}")
        return parts[1], "", float(parts[0])

    raise ValueError(f"Unsupported workload format: {workload_format}")


def _detect_format(parts: list[str]) -> WorkloadFormat:
    if len(parts) >= 2 and _looks_numeric(parts[0]) and _looks_like_select(parts[1]):
        return "complete"
    if len(parts) >= 3 and _looks_like_select(parts[0]) and _looks_numeric(parts[2]):
        return "single_table"
    if len(parts) >= 2 and _looks_like_select(parts[0]):
        return "subquery"
    raise ValueError(f"Could not detect STATS-CEB workload format from fields: {parts!r}")


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _looks_like_select(value: str) -> bool:
    return value.lower().startswith("select")


def _normalize_label(value: str) -> str:
    label = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    return label or "stats_ceb"

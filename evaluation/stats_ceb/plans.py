from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


_VOLATILE_EXACT = {
    "Planning Time",
    "Execution Time",
    "Actual Startup Time",
    "Actual Total Time",
    "Actual Rows",
    "Actual Loops",
    "Rows Removed by Filter",
    "Rows Removed by Index Recheck",
    "Heap Fetches",
    "Workers",
}
_VOLATILE_PREFIXES = (
    "Shared ",
    "Local ",
    "Temp ",
    "I/O ",
    "WAL ",
)
_HINT_PATTERN = re.compile(r"/\*=pg_lab=.*?\*/", re.IGNORECASE | re.DOTALL)


def explain_json(
    database: Any,
    query: Any,
    *,
    analyze: bool = False,
    buffers: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Obtains PostgreSQL JSON through PostBOUND's Database interface."""
    options = ["FORMAT JSON"]
    if analyze:
        options.append("ANALYZE TRUE")
    if buffers:
        options.append("BUFFERS TRUE")
    sql = str(query).strip().rstrip(";")
    explain_sql = f"EXPLAIN ({', '.join(options)}) {sql}"
    result = database.execute_query(
        explain_sql,
        cache_enabled=False,
        raw=True,
        timeout=timeout_seconds,
    )
    return normalize_explain_result(result)


def normalize_explain_result(result: Any) -> dict[str, Any]:
    value = result
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, dict) for item in value):
        merged: dict[str, Any] = {}
        for item in value:
            merged.update(item)
        value = merged
    if not isinstance(value, dict) or "Plan" not in value:
        raise ValueError(f"Unexpected EXPLAIN FORMAT JSON result shape: {type(value).__name__}")
    return _json_safe(value)


def plan_root(document: dict[str, Any]) -> dict[str, Any]:
    root = document.get("Plan")
    if not isinstance(root, dict):
        raise ValueError("EXPLAIN JSON document does not contain a plan root")
    return root


def root_plan_rows(document: dict[str, Any]) -> float:
    value = float(plan_root(document)["Plan Rows"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid root Plan Rows: {value!r}")
    return value


def canonicalize_plan(document: dict[str, Any]) -> dict[str, Any]:
    return _canonicalize_value(plan_root(document))


def stable_plan_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(
        canonicalize_plan(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_plan(path: str | Path, document: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return target


def save_sql(path: str | Path, query: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(query).rstrip() + "\n", encoding="utf-8")
    return target


def extract_hint(sql_or_query: Any) -> str:
    match = _HINT_PATTERN.search(str(sql_or_query))
    return "" if match is None else match.group(0)


def extract_base_scans(document: dict[str, Any]) -> list[dict[str, Any]]:
    scans: list[dict[str, Any]] = []
    for node in iter_plan_nodes(plan_root(document), postorder=False):
        relation = node.get("Relation Name")
        if not relation:
            continue
        scans.append(
            {
                "relation_key": relation_key(str(relation), str(node.get("Alias") or "")),
                "relation_name": str(relation),
                "alias": str(node.get("Alias") or ""),
                "node_type": str(node.get("Node Type") or ""),
                "plan_rows": _finite_optional(node.get("Plan Rows")),
                "filter": node.get("Filter"),
                "index_cond": node.get("Index Cond"),
            }
        )
    return scans


def extract_joins(document: dict[str, Any]) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for node in iter_plan_nodes(plan_root(document), postorder=True):
        if not _is_join(node):
            continue
        branches = [_leaf_relations(child) for child in _children(node)]
        relations = sorted(set().union(*branches)) if branches else []
        is_base = len(branches) == 2 and all(len(branch) == 1 for branch in branches)
        key = canonical_join_key(relations) if is_base else None
        joins.append(
            {
                "node_type": str(node.get("Node Type") or ""),
                "join_type": node.get("Join Type"),
                "relations": relations,
                "branch_relations": [sorted(branch) for branch in branches],
                "is_base_join": is_base,
                "base_join_key": key,
                "plan_rows": _finite_optional(node.get("Plan Rows")),
            }
        )
    return joins


def all_base_join_keys(document: dict[str, Any]) -> list[str]:
    return [str(join["base_join_key"]) for join in extract_joins(document) if join["is_base_join"]]


def first_base_join_key(document: dict[str, Any]) -> str | None:
    joins = all_base_join_keys(document)
    return joins[0] if joins else None


def canonical_join_key(relations: Iterable[str]) -> str:
    return "|".join(sorted(set(relations)))


def relation_key(relation_name: str, alias: str = "") -> str:
    return f"{relation_name}:{alias}" if alias and alias != relation_name else relation_name


def iter_plan_nodes(root: dict[str, Any], *, postorder: bool) -> Iterable[dict[str, Any]]:
    if not postorder:
        yield root
    for child in _children(root):
        yield from iter_plan_nodes(child, postorder=postorder)
    if postorder:
        yield root


def analyze_metrics(document: dict[str, Any]) -> dict[str, Any]:
    root = plan_root(document)
    return {
        "planning_time_ms": _finite_optional(document.get("Planning Time")),
        "execution_time_ms": _finite_optional(document.get("Execution Time")),
        "actual_rows": _finite_optional(root.get("Actual Rows")),
        "actual_loops": _finite_optional(root.get("Actual Loops")),
        "shared_hit_blocks": _sum_plan_metric(root, "Shared Hit Blocks"),
        "shared_read_blocks": _sum_plan_metric(root, "Shared Read Blocks"),
        "shared_dirtied_blocks": _sum_plan_metric(root, "Shared Dirtied Blocks"),
        "shared_written_blocks": _sum_plan_metric(root, "Shared Written Blocks"),
        "temp_read_blocks": _sum_plan_metric(root, "Temp Read Blocks"),
        "temp_written_blocks": _sum_plan_metric(root, "Temp Written Blocks"),
    }


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("Plans", [])
    return [child for child in children if isinstance(child, dict)] if isinstance(children, list) else []


def _is_join(node: dict[str, Any]) -> bool:
    node_type = str(node.get("Node Type") or "")
    return "Join" in node_type or node_type == "Nested Loop" or "Join Type" in node


def _leaf_relations(node: dict[str, Any]) -> set[str]:
    relation = node.get("Relation Name")
    if relation:
        return {relation_key(str(relation), str(node.get("Alias") or ""))}
    relations: set[str] = set()
    for child in _children(node):
        relations.update(_leaf_relations(child))
    return relations


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key in _VOLATILE_EXACT or key.startswith(_VOLATILE_PREFIXES):
                continue
            output[key] = _canonicalize_value(item)
        return output
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _sum_plan_metric(root: dict[str, Any], key: str) -> float:
    # PostgreSQL reports inclusive buffer counters at the root, so do not double count children.
    value = root.get(key, 0)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0

from __future__ import annotations

from collections.abc import Sequence

from postbound import TableReference, qal as pb_qal

from core.domain import InternalPredicate, PredicateOperator, TranslationResult, ensure_tuple
from .qal_utils import qualified_table_name

_SUPPORTED_OPERATORS = {
    "=": PredicateOperator.EQ,
    "<": PredicateOperator.LT,
    "<=": PredicateOperator.LE,
    ">": PredicateOperator.GT,
    ">=": PredicateOperator.GE,
    "between": PredicateOperator.BETWEEN,
    "in": PredicateOperator.IN,
}


def extract_request_tables(
    intermediate: TableReference | Sequence[TableReference] | None,
) -> list[TableReference]:
    if intermediate is None:
        return []
    if isinstance(intermediate, TableReference):
        return [intermediate]
    return list(intermediate)


def resolve_table_name(table: TableReference) -> str:
    return qualified_table_name(table)


def base_filter_predicate(
    query: pb_qal.SqlQuery,
    table: TableReference,
) -> pb_qal.AbstractPredicate | None:
    predicates = query.predicates()
    if predicates is None:
        return None
    relevant = predicates.filters_for(table)
    if relevant is not None:
        return relevant
    raw_filters = list(predicates.filters())
    if not raw_filters:
        return None
    query_tables = list(query.tables())
    if len(query_tables) == 1:
        return pb_qal.CompoundPredicate.create_and(raw_filters)
    return None


def _simple_filters_for(
    query: pb_qal.SqlQuery,
    table: TableReference,
) -> list[pb_qal.SimpleFilter]:
    predicates = query.predicates()
    if predicates is None:
        return []
    raw_filters = list(predicates.filters())
    if not raw_filters:
        return []
    query_tables = list(query.tables())
    if len(query_tables) == 1:
        relevant = raw_filters
    else:
        relevant = [predicate for predicate in raw_filters if predicate.contains_table(table)]

    translated: list[pb_qal.SimpleFilter] = []
    for predicate in relevant:
        try:
            translated.append(pb_qal.SimpleFilter(predicate))
        except ValueError as exc:
            raise ValueError("complex boolean predicates are unsupported") from exc
    return translated


def _normalize_operator(raw_operator: object) -> str:
    if raw_operator is None:
        raise ValueError("Missing predicate operator")
    value = getattr(raw_operator, "value", None)
    if value is not None:
        return str(value).lower()
    return str(raw_operator).lower()


def _normalize_value(raw_value: object) -> object:
    value = getattr(raw_value, "value", None)
    return raw_value if value is None else value


def _column_name(column: pb_qal.ColumnReference) -> str:
    return str(column.name)


def _column_table_name(column: pb_qal.ColumnReference) -> str | None:
    if column.table is None:
        return None
    return resolve_table_name(column.table)


def _convert_predicate(simple_filter: pb_qal.SimpleFilter, table_name: str) -> InternalPredicate:
    column = simple_filter.column
    resolved_table = _column_table_name(column)
    if resolved_table is not None and resolved_table != table_name:
        raise ValueError("Cross-table predicate detected")

    normalized_op = _normalize_operator(simple_filter.operation)
    if normalized_op not in _SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported operator: {normalized_op}")
    op = _SUPPORTED_OPERATORS[normalized_op]
    column_name = _column_name(column)

    if op == PredicateOperator.BETWEEN:
        lower, upper = ensure_tuple(simple_filter.value)
        return InternalPredicate(table_name, column_name, op, lower=lower, upper=upper)
    if op == PredicateOperator.IN:
        normalized_values = tuple(_normalize_value(value) for value in ensure_tuple(simple_filter.value))
        return InternalPredicate(table_name, column_name, op, values=normalized_values)

    value = _normalize_value(simple_filter.value)
    return InternalPredicate(table_name, column_name, op, value=value)


def translate_request(
    query: pb_qal.SqlQuery,
    intermediate: TableReference | Sequence[TableReference] | None,
) -> TranslationResult:
    tables = extract_request_tables(intermediate)
    if len(tables) != 1:
        return TranslationResult(table=None, predicates=[], fallback_reason="request is not single-table")

    table_name = resolve_table_name(tables[0])

    translated: list[InternalPredicate] = []
    try:
        for simple_filter in _simple_filters_for(query, tables[0]):
            translated.append(_convert_predicate(simple_filter, table_name))
    except Exception as exc:
        return TranslationResult(table=None, predicates=[], fallback_reason=str(exc))
    return TranslationResult(table=tables[0], predicates=translated)

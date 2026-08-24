from __future__ import annotations

from collections.abc import Sequence

from postbound import ColumnReference, TableReference, qal as pb_qal


def qualified_table_name(table: TableReference) -> str:
    return table.qualified_name() if table.schema else table.full_name


def build_table_query(
    table: TableReference,
    *,
    columns: Sequence[ColumnReference] | None = None,
    predicate: pb_qal.AbstractPredicate | None = None,
    limit: int | None = None,
    count_star: bool = False,
) -> pb_qal.SqlQuery:
    if count_star:
        select_clause = pb_qal.Select(
            pb_qal.BaseProjection(
                pb_qal.FunctionExpression("COUNT", [pb_qal.StarExpression()])
            )
        )
    elif columns is None:
        select_clause = pb_qal.Select(pb_qal.BaseProjection.star())
    else:
        select_clause = pb_qal.Select(
            [
                pb_qal.BaseProjection(pb_qal.ColumnExpression(column))
                for column in columns
            ]
        )

    clauses: list[pb_qal.BaseClause] = [
        select_clause,
        pb_qal.ImplicitFromClause(pb_qal.DirectTableSource(table)),
    ]
    if predicate is not None:
        clauses.append(pb_qal.Where(predicate))
    if limit is not None:
        clauses.append(pb_qal.Limit(limit=int(limit)))
    return pb_qal.build_query(clauses)

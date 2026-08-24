from __future__ import annotations

import pandas as pd

from core.config import NaruConfig
from core.domain import ColumnInfo, ColumnKind, InternalPredicate, PredicateOperator
from core.domain import TableInfo
from model.encoding.encoders import ColumnEncoder, PredicateApplicationError, TableEncoder


def test_dictionary_encoder_supports_in_predicate() -> None:
    config = NaruConfig()
    column = ColumnInfo("public", "items", "category", "text", ColumnKind.TEXT, 1, False, True)
    encoder = ColumnEncoder.fit(column, pd.Series(["a", "b", "c", "a"]), config)
    predicate = InternalPredicate("items", "category", PredicateOperator.IN, values=("a", "c"))
    allowed = encoder.allowed_ids(predicate)
    assert len(allowed) == 2


def test_table_encoder_intersects_multiple_predicates_on_same_column() -> None:
    config = NaruConfig()
    table_info = TableInfo(
        schema_name="public",
        table_name="items",
        estimated_row_count=3,
        exact_row_count=3,
        columns=[ColumnInfo("public", "items", "value", "integer", ColumnKind.INTEGER, 1, False, True)],
    )
    encoder, _ = TableEncoder.fit(table_info, pd.DataFrame({"value": [1, 2, 3]}), config)

    allowed = encoder.allowed_token_ids(
        [
            InternalPredicate("public.items", "value", PredicateOperator.GE, value=2),
            InternalPredicate("public.items", "value", PredicateOperator.LE, value=2),
        ]
    )

    assert allowed[0] == [1]


def test_table_encoder_rejects_untrained_columns() -> None:
    config = NaruConfig()
    table_info = TableInfo(
        schema_name="public",
        table_name="items",
        estimated_row_count=1,
        exact_row_count=1,
        columns=[ColumnInfo("public", "items", "category", "text", ColumnKind.TEXT, 1, False, True)],
    )
    encoder, _ = TableEncoder.fit(table_info, pd.DataFrame({"category": ["a"]}), config)

    try:
        encoder.allowed_token_ids(
            [InternalPredicate("public.items", "missing", PredicateOperator.EQ, value="a")]
        )
    except PredicateApplicationError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover - defensive branch for pytest-style tests
        raise AssertionError("Expected a PredicateApplicationError for an untrained column")


def test_dictionary_range_predicates_ignore_null_sentinel_values() -> None:
    config = NaruConfig(numeric_max_unique_for_dictionary=10)
    column = ColumnInfo("public", "items", "score", "integer", ColumnKind.INTEGER, 1, True, True)
    encoder = ColumnEncoder.fit(column, pd.Series([None, 10, 20]), config)

    allowed = encoder.allowed_ids(
        InternalPredicate("public.items", "score", PredicateOperator.GE, value=15)
    )

    assert len(allowed) == 1
    assert encoder.inverse_dictionary[allowed[0]] == 20.0

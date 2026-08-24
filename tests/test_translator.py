from __future__ import annotations

import postbound as pb

from integration.postbound.translator import translate_request


def test_translator_accepts_postbound_style_query() -> None:
    request = pb.parse_query("SELECT * FROM orders WHERE status = 'shipped' AND total >= 100")
    orders = next(iter(request.tables()))

    translated = translate_request(request, orders)

    assert translated.can_estimate
    assert translated.table_name == "orders"
    assert len(translated.predicates) == 2
    assert {predicate.column_name for predicate in translated.predicates} == {"status", "total"}


def test_translator_rejects_multi_table_intermediate() -> None:
    request = pb.parse_query("SELECT * FROM orders, customers")
    tables = list(request.tables())

    translated = translate_request(request, tables)

    assert not translated.can_estimate
    assert translated.fallback_reason == "request is not single-table"


def test_translator_rejects_complex_boolean_predicates() -> None:
    request = pb.parse_query("SELECT * FROM orders WHERE status = 'shipped' OR total >= 100")
    orders = next(iter(request.tables()))

    translated = translate_request(request, orders)

    assert not translated.can_estimate
    assert translated.fallback_reason == "complex boolean predicates are unsupported"

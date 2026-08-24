from __future__ import annotations

from dataclasses import dataclass

import postbound as pb

from core.domain import EstimationResult
from integration.postbound.estimator import PostboundCardinalityEstimator


class FakeRegistry:
    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def has_model(self, table_name: str) -> bool:
        return table_name == "orders"

    def available_tables(self) -> list[str]:
        return ["orders"]

    def estimate(self, table_name: str, predicates: list[object]) -> EstimationResult:
        self.seen.append((table_name, len(predicates)))
        return EstimationResult(
            table_name=table_name,
            selectivity=0.125,
            cardinality=42.0,
            used_model=True,
        )


class FakeNativeEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def cardinality_estimate(self, _query: object) -> float:
        return self.value


class FakeTargetDb:
    def __init__(self, value: float) -> None:
        self._optimizer = FakeNativeEstimator(value)

    def optimizer(self) -> FakeNativeEstimator:
        return self._optimizer


def test_postbound_estimator_calculates_base_table_cardinality() -> None:
    query = pb.parse_query("SELECT * FROM orders WHERE status = 'shipped'")
    orders = next(iter(query.tables()))
    registry = FakeRegistry()
    estimator = PostboundCardinalityEstimator(registry=registry)
    estimator.initialize(FakeTargetDb(0.0), query)

    cardinality = estimator.calculate_estimate(query, orders)

    assert cardinality == 42.0
    assert registry.seen == [("orders", 1)]


def test_postbound_estimator_delegates_unsupported_intermediate_to_fallback() -> None:
    query = pb.parse_query("SELECT * FROM orders")
    orders = next(iter(query.tables()))
    customers = pb.TableReference("customers")
    estimator = PostboundCardinalityEstimator(registry=FakeRegistry())
    estimator.initialize(FakeTargetDb(13.0), query)

    cardinality = estimator.calculate_estimate(query, [orders, customers])

    assert cardinality == 13.0


def test_postbound_query_objects_work_with_estimator() -> None:
    query = pb.parse_query("SELECT * FROM orders WHERE status = 'shipped'")
    table = next(iter(query.tables()))
    estimator = PostboundCardinalityEstimator(registry=FakeRegistry())
    estimator.initialize(FakeTargetDb(0.0), query)

    assert estimator.calculate_estimate(query, table) == 42.0

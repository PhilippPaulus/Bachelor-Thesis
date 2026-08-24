from __future__ import annotations

import math
import csv

import postbound as pb

from evaluation.stats_ceb.cardinality import BaseOnlyCardinalityParameterization, ExactCardinalityCache, canonical_join_key, cardinality_target_query
from evaluation.stats_ceb.failure_analysis import selectivity_bucket
from evaluation.stats_ceb.metrics import geometric_mean, paired_bootstrap_ci, paired_cluster_bootstrap_ci, percentile, q_error
from evaluation.stats_ceb.plans import all_base_join_keys, canonicalize_plan, first_base_join_key, root_plan_rows, stable_plan_hash
from evaluation.stats_ceb.plots import generate_all_plots
from evaluation.stats_ceb.single_table import single_table_summary
from evaluation.stats_ceb.workload import load_stats_ceb_workload
from scripts.experiment_1_accuracy import COMPLETE_FIELDS
from scripts.experiment_4_base_join_quality import FIELDS as QUALITY_FIELDS, build_summary as build_quality_summary, classify_decision
from scripts.experiment_5_runtime_impact import (
    PER_QUERY_FIELDS,
    REPETITION_FIELDS,
    build_per_query,
    build_summary as build_runtime_summary,
    execution_order_for_repetition,
)


class FakeDb:
    def __init__(self, value: float = 42.0) -> None:
        self.value = value
        self.executed: list[object] = []

    def execute_query(self, query: object, **_: object) -> list[tuple[float]]:
        self.executed.append(query)
        return [(self.value,)]


def test_stats_ceb_loader_parses_all_supported_formats(tmp_path) -> None:
    single = tmp_path / "single.sql"
    single.write_text("SELECT COUNT(*) FROM badges as b;||0||79851\n", encoding="utf-8")
    subquery = tmp_path / "subquery.sql"
    subquery.write_text("SELECT COUNT(*) FROM users as u, badges as b WHERE b.UserId = u.Id;||0\n", encoding="utf-8")
    complete = tmp_path / "stats_CEB.sql"
    complete.write_text("79851||SELECT COUNT(*) FROM badges as b, users as u WHERE b.UserId = u.Id;\n", encoding="utf-8")

    single_rows = load_stats_ceb_workload(single, workload_format="single_table")
    subquery_rows = load_stats_ceb_workload(subquery, workload_format="subquery")
    complete_rows = load_stats_ceb_workload(complete, workload_format="complete")

    assert single_rows[0].actual_cardinality == 79851.0
    assert single_rows[0].query_id == "0"
    assert subquery_rows[0].actual_cardinality is None
    assert subquery_rows[0].query_size == 2
    assert complete_rows[0].actual_cardinality == 79851.0
    assert complete_rows[0].sql.startswith("SELECT COUNT(*)")
    assert complete_rows[0].label == "stats_ceb_00001"
    assert len(complete_rows[0].normalized_sql_id) == 64


def test_metrics_and_paired_cluster_bootstrap_are_explicit() -> None:
    assert q_error(10, 5) == 2
    assert q_error(0.2, 0) == 1
    assert percentile([1, 2, 3], 0.5) == 2
    assert math.isclose(geometric_mean([1, 4]), 2.0)

    payload = paired_bootstrap_ci([1, 2, 3], [2, 2, 2], lambda values: sum(values) / len(values), samples=25)
    clustered = paired_cluster_bootstrap_ci(
        [1, 2, 3], [2, 2, 2], ["duplicate", "duplicate", "unique"],
        lambda values: sum(values) / len(values), samples=25,
    )

    assert payload["sample_count"] == 3
    assert payload["bootstrap_samples"] == 25
    assert payload["cluster_count"] is None
    assert clustered["sample_count"] == 3
    assert clustered["cluster_count"] == 2


def test_base_only_parameterization_emits_singleton_positive_hints_only() -> None:
    query = pb.parse_query("SELECT COUNT(*) FROM badges as b, users as u WHERE b.UserId = u.Id")
    parameterization = BaseOnlyCardinalityParameterization(
        "exact_base",
        FakeDb(0.0),
        exact_cache=ExactCardinalityCache(FakeDb(0.0)),
    ).generate_plan_parameters(query, None, None)

    assert parameterization.cardinalities
    assert all(len(tables) == 1 for tables in parameterization.cardinalities)
    assert all(value == 1.0 for value in parameterization.cardinalities.values())


def test_complete_cardinality_target_query_drops_count_and_avoids_aggregate_root() -> None:
    query = pb.parse_query("SELECT COUNT(*) FROM badges as b, users as u WHERE b.UserId = u.Id")
    target = cardinality_target_query(query)
    aggregate_document = {"Plan": {"Node Type": "Aggregate", "Plan Rows": 1, "Plans": [{"Node Type": "Hash Join", "Plan Rows": 123}]}}

    assert str(target).lower().startswith("select *")
    assert "count" not in str(target).lower()
    assert root_plan_rows(aggregate_document) == 1
    assert str(target) != str(query)


def test_plan_canonicalization_hash_ignores_runtime_but_retains_estimates() -> None:
    first = {
        "Plan": {
            "Node Type": "Seq Scan", "Relation Name": "posts", "Alias": "p", "Plan Rows": 42,
            "Total Cost": 10.0, "Actual Rows": 41, "Actual Total Time": 2.0, "Shared Hit Blocks": 8,
        },
        "Planning Time": 1.0,
        "Execution Time": 3.0,
    }
    second = {
        "Plan": {
            "Node Type": "Seq Scan", "Relation Name": "posts", "Alias": "p", "Plan Rows": 42,
            "Total Cost": 10.0, "Actual Rows": 999, "Actual Total Time": 20.0, "Shared Hit Blocks": 80,
        },
        "Planning Time": 10.0,
        "Execution Time": 30.0,
    }
    changed_estimate = {"Plan": {**second["Plan"], "Plan Rows": 43}}

    assert stable_plan_hash(first) == stable_plan_hash(second)
    assert stable_plan_hash(first) != stable_plan_hash(changed_estimate)
    assert "Actual Rows" not in canonicalize_plan(first)
    assert canonicalize_plan(first)["Plan Rows"] == 42


def test_base_join_extraction_handles_unary_wrappers_and_canonical_keys() -> None:
    document = {
        "Plan": {
            "Node Type": "Gather", "Plan Rows": 10,
            "Plans": [{
                "Node Type": "Nested Loop", "Plan Rows": 10,
                "Plans": [
                    {"Node Type": "Materialize", "Plans": [{"Node Type": "Seq Scan", "Relation Name": "posts", "Alias": "p", "Plan Rows": 5}]},
                    {"Node Type": "Index Scan", "Relation Name": "badges", "Alias": "b", "Plan Rows": 2},
                ],
            }],
        }
    }

    assert first_base_join_key(document) == "badges:b|posts:p"
    assert canonical_join_key({pb.TableReference("posts", alias="p"), pb.TableReference("badges", alias="b")}) == "badges:b|posts:p"


def test_bushy_plan_records_all_base_joins_and_outer_prioritized_first() -> None:
    document = {
        "Plan": {
            "Node Type": "Hash Join", "Plan Rows": 10,
            "Plans": [
                {"Node Type": "Nested Loop", "Plans": [
                    {"Node Type": "Seq Scan", "Relation Name": "a", "Alias": "a"},
                    {"Node Type": "Seq Scan", "Relation Name": "b", "Alias": "b"},
                ]},
                {"Node Type": "Merge Join", "Plans": [
                    {"Node Type": "Seq Scan", "Relation Name": "c", "Alias": "c"},
                    {"Node Type": "Seq Scan", "Relation Name": "d", "Alias": "d"},
                ]},
            ],
        }
    }

    assert all_base_join_keys(document) == ["a|b", "c|d"]
    assert first_base_join_key(document) == "a|b"


def test_exact_join_cardinality_is_acquired_even_when_cached() -> None:
    query = pb.parse_query("SELECT * FROM badges AS b, users AS u WHERE b.userid = u.id")
    cache = ExactCardinalityCache(FakeDb(17.0))
    first = cache.measure_intermediate(query, frozenset(query.tables()))
    second = cache.measure_intermediate(query, frozenset(query.tables()))

    assert first.cardinality == 17.0
    assert first.status == "executed"
    assert "COUNT" in first.sql
    assert second.status == "cached"
    assert second.cardinality == 17.0


def test_single_table_summary_separates_occurrences_and_unique_sql() -> None:
    base = {
        "status": "ok", "is_filtered": True, "used_model": True, "fallback_used": False,
        "learned_q_error": 1.0, "native_q_error": 2.0,
        "learned_signed_error_ratio": 1.0, "native_signed_error_ratio": 2.0,
    }
    rows = [
        {**base, "normalized_sql_id": "same"},
        {**base, "normalized_sql_id": "same"},
        {**base, "normalized_sql_id": "other", "learned_q_error": 4.0},
    ]

    summary = single_table_summary(rows)

    assert summary["sections"]["all_workload_occurrences"]["count"] == 3
    assert summary["sections"]["unique_normalized_sql"]["count"] == 2
    assert summary["sections"]["filtered_workload_occurrences"]["unique_query_count"] == 2


def test_exact_base_decision_categories_use_no_regret_placeholder() -> None:
    assert classify_decision("a|b", "b|c", "b|c") == "improved"
    assert classify_decision("a|b", "b|c", "a|b") == "degraded"
    assert classify_decision("a|b", "a|b", "a|b") == "both agree"
    assert classify_decision("a|b", "a|b", "b|c") == "neither agrees"
    assert not any("regret" in field for field in QUALITY_FIELDS)

    invalid = [{
        "status": "ok", "treatment_valid": True,
        "native_agrees_exact_base_reference": True, "learned_agrees_exact_base_reference": True,
        "decision_category": "both agree", "native_relative_first_join_output": math.nan,
        "learned_relative_first_join_output": 1.0,
    }]
    try:
        build_quality_summary(invalid, samples=10)
    except ValueError:
        pass
    else:
        raise AssertionError("Experiment 4 summary accepted NaN")


def test_runtime_order_alternates_within_each_repetition() -> None:
    configs = ["native", "learned_base", "exact_base"]
    assert execution_order_for_repetition(configs, 0) == ["native", "learned_base", "exact_base"]
    assert execution_order_for_repetition(configs, 1) == ["learned_base", "native", "exact_base"]
    assert execution_order_for_repetition(configs, 2) == ["native", "learned_base", "exact_base"]


def test_runtime_timeout_accounting_and_censoring() -> None:
    common = {
        "query_id": "q1", "source_query_id": "1", "query_template": "t", "relation_count": 2,
        "warmup": False, "repetition": 0, "execution_order": 1, "base_join_changed": True,
        "full_plan_changed": True, "treatment_category": "base join changed", "base_join": "a|b",
        "plan_hash": "hash", "wall_clock_duration_ms": 100.0, "treatment_valid": True,
    }
    rows = [
        {**common, "configuration": "native", "execution_time_ms": 100.0, "timeout": False, "status": "ok"},
        {**common, "configuration": "learned_base", "execution_time_ms": None, "timeout": True, "status": "timeout"},
    ]

    per_query = build_per_query(rows, timeout_seconds=300.0)
    summary = build_runtime_summary(rows, per_query_rows=per_query, timeout_seconds=300.0, samples=10)

    assert per_query[0]["learned_median_execution_time_ms"] == 300_000.0
    assert per_query[0]["learned_timeout_count"] == 1
    assert summary["timeout_counts"]["learned_base"] == 1
    assert summary["runtime_classifications"]["degraded"] == 1


def test_artifact_schemas_include_validity_and_audit_columns() -> None:
    assert {
        "aggregate_free_sql", "native_plan_hash", "learned_plan_hash", "exact_base_plan_hash",
        "preflight_passed", "hint_roundtrip_valid", "treatment_valid",
    }.issubset(COMPLETE_FIELDS)
    assert {
        "native_exact_join_sql", "learned_exact_join_sql", "exact_base_reference_exact_join_sql",
        "missing_value_status", "native_relative_first_join_output", "learned_relative_first_join_output",
    }.issubset(QUALITY_FIELDS)
    assert {"execution_order", "execution_time_ms", "planning_time_ms", "timeout", "treatment_valid"}.issubset(REPETITION_FIELDS)
    assert {"speedup_native_over_learned", "runtime_classification"}.issubset(PER_QUERY_FIELDS)


def test_documented_selectivity_buckets() -> None:
    assert selectivity_bucket(0) == "exactly_zero"
    assert selectivity_bucket(0.0001) == "(0,0.0001]"
    assert selectivity_bucket(0.001) == "(0.0001,0.001]"
    assert selectivity_bucket(1.0) == "(0.5,1]"


def test_csv_driven_plot_generation_covers_all_experiments(tmp_path) -> None:
    _csv(tmp_path / "experiment_1_accuracy" / "single_table_results.csv", [{
        "status": "ok", "native_q_error": 2.0, "learned_q_error": 1.5,
    }])
    _csv(tmp_path / "experiment_1_accuracy" / "complete_query_results.csv", [{
        "status": "ok", "native_q_error": 4.0, "learned_base_q_error": 2.0,
        "learned_improvement_ratio": 2.0, "exact_base_attainable_improvement_ratio": 3.0,
    }])
    _csv(tmp_path / "experiment_2_failures" / "grouped_failures.csv", [
        {"group_dimension": "predicate_count", "group_value": "2", "learned_median_q_error": 2.0},
        {"group_dimension": "selectivity_bucket", "group_value": "(0,0.0001]", "learned_median_q_error": 3.0},
        {"group_dimension": "table", "group_value": "posts", "learned_median_q_error": 4.0},
        {"group_dimension": "constrained_column", "group_value": "posts.score", "learned_median_q_error": 5.0},
    ])
    _csv(tmp_path / "experiment_2_failures" / "worst_queries.csv", [{"learned_signed_error_ratio": 10.0}])
    _csv(tmp_path / "experiment_3_base_join_influence" / "base_join_changes.csv", [{
        "applicable": True, "base_join_changed_native_learned": True, "relation_count": 5,
        "maximum_absolute_log_cardinality_difference": 2.0,
    }])
    _csv(tmp_path / "experiment_4_base_join_quality" / "decision_quality.csv", [{
        "status": "ok", "decision_category": "improved", "native_relative_first_join_output": 3.0,
        "learned_relative_first_join_output": 1.0,
    }])
    _csv(tmp_path / "experiment_5_runtime" / "runtime_per_query.csv", [{
        "status": "complete", "native_median_execution_time_ms": 20.0,
        "learned_median_execution_time_ms": 10.0, "speedup_native_over_learned": 2.0,
        "full_plan_changed": True,
    }])

    generated = generate_all_plots(tmp_path)

    assert len(generated) == 17
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)


def _csv(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

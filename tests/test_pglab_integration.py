from __future__ import annotations

import os
from pathlib import Path

import postbound as pb
import pytest

from backends.postgres import connect_postgres_database
from evaluation.stats_ceb.plans import explain_json, root_plan_rows
from evaluation.stats_ceb.preflight import run_preflight
from evaluation.stats_ceb.run_context import RunContext
from evaluation.stats_ceb.workload import StatsCebQuery
from registry.registry import ModelRegistry


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("RUN_PGLAB_INTEGRATION") != "1", reason="set RUN_PGLAB_INTEGRATION=1")
def test_real_pglab_roundtrip_and_stats_ceb_00135_treatments(tmp_path) -> None:
    connection_file = Path(os.environ.get("PGLAB_CONNECTION_FILE", ".psycopg_connection_stats_pglab")).resolve()
    model_dir = Path(os.environ.get("STATS_MODEL_DIR", "artifacts/models/stats")).resolve()
    database = connect_postgres_database(connection_file.read_text(encoding="utf-8").strip(), cache_enabled=False, private=True)
    registry = ModelRegistry.load(model_dir)
    registry.configure_inference(random_seed=42)

    native = explain_json(database, pb.parse_query("SELECT * FROM posts AS p WHERE p.score > 10"))
    hinted = explain_json(database, pb.parse_query("/*=pg_lab= Card(p #42) */ SELECT * FROM posts AS p WHERE p.score > 10"))
    assert root_plan_rows(native) != 42
    assert root_plan_rows(hinted) == 42

    sql = (
        "SELECT COUNT(*) FROM comments as c, posts as p, postHistory as ph, badges as b, users as u "
        "WHERE u.Id = ph.UserId AND u.Id = b.UserId AND u.Id = p.OwnerUserId AND u.Id = c.UserId "
        "AND c.Score=0 AND p.PostTypeId=1 AND p.ViewCount>=0 AND p.ViewCount<=4157 "
        "AND p.FavoriteCount=0 AND p.CreationDate<='2014-09-08 09:58:16'::timestamp;"
    )
    query = pb.parse_query(sql)
    item = StatsCebQuery(
        label="stats_ceb_00135", sql=sql, query=query, query_id="135", line_number=135,
        query_size=5, template="stats_ceb", actual_cardinality=2263957167.0,
    )
    context = RunContext.create(tmp_path, "integration")
    payload = run_preflight(
        context,
        database=database,
        registry=registry,
        sample_query=item,
        injection_validation=True,
        timeout_seconds=300.0,
        expected_exact={"posts:p": 879.0},
    )

    project = payload["project_generated_treatment"]
    configurations = project["configurations"]
    assert payload["preflight_passed"]
    assert configurations["native"]["hint"] == ""
    assert "Card(" in configurations["learned_base"]["hint"]
    assert "Card(" in configurations["exact_base"]["hint"]
    assert configurations["exact_base"]["base_estimates"]["posts:p"] == 879.0
    assert project["aggregate_removed"]
    assert all((context.run_dir / configurations[name]["plan_path"]).exists() for name in configurations)
    assert all(configurations[name]["plan_hash"] for name in configurations)
    assert project["standalone_relation_roundtrips"]["learned_base"]
    assert project["standalone_relation_roundtrips"]["exact_base"]

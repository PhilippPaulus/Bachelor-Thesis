# Naru–PostBOUND Integration

Prototype implementation for the bachelor thesis **Improving Base Join Selection with Precise Single-Table Cardinality Estimates**.

The project trains one Naru-style MADE autoregressive density model per supported base table and integrates the resulting single-table cardinality estimates into PostBOUND. Learned estimates are supplied only for singleton base relations; PostgreSQL remains responsible for join-cardinality estimation, join ordering, costing, and physical-operator selection.

## Repository structure

- `core/` — shared configuration and domain/type definitions
- `model/encoding/` — relational value encoding and numeric discretization
- `model/naru/` — MADE model, training, persistence, enumeration, and progressive-sampling inference
- `training/` — per-table training pipeline and command-line entry point
- `registry/` — model discovery, lazy loading, and estimate caching
- `integration/postbound/` — predicate translation and PostBOUND cardinality-estimator integration
- `backends/postgres/` — PostgreSQL/PostBOUND connection adapter
- `evaluation/stats_ceb/` — reusable STATS-CEB evaluation logic
- `scripts/` — thesis experiment entry points, preflight validation, orchestration, and plot generation
- `tests/` — unit and integration tests
- `artifacts/evaluations/README.md` — schema and interpretation of generated evaluation artifacts

## Environment

The thesis evaluation used Python 3.12, PostBOUND 0.21.5, PostgreSQL with `pg_lab`, and PyTorch 2.6.0. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build appropriate for the local CUDA environment if it differs from the default package installation.

A reachable PostgreSQL instance containing the target workload is required. The optimizer experiments additionally require `pg_lab` because base-table cardinalities are injected through `Card(...)` hints.

## Training

Train all supported tables in a schema:

```bash
python -m training.train_models \
  --conn-string "<postgres-connection-string>" \
  --schema public \
  --output-dir artifacts/models/stats
```

Use `--table <table-name>` to train only one table. Other architecture and training parameters can be overridden through the command-line options; their defaults come from `core/config.py`.

## Evaluation

The complete STATS-CEB evaluation can be run through the orchestration script:

```bash
python scripts/run_all_experiments.py \
  --conn-string "<postgres-connection-string>" \
  --model-dir artifacts/models/stats \
  --output-root artifacts/evaluations \
  --run-id <run-id> \
  --complete-workload-path <path-to-stats-ceb-workload>
```

The orchestration first runs the preflight validation and then executes the five thesis experiments in order. Generated evaluation runs are intentionally ignored by Git; their file layout and statistical definitions are documented in `artifacts/evaluations/README.md`.

## Tests

Run the regular test suite with:

```bash
pytest
```

The real PostgreSQL/`pg_lab` integration test is marked separately and requires the corresponding database, model artifacts, and environment configuration.

## Scope

This repository is research prototype code for studying the downstream optimizer effect of learned **single-table** cardinality estimates. It is not a replacement query optimizer and does not learn or inject join-cardinality estimates.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from core.config import NaruConfig
from core.domain import TableArtifacts
from ..encoding.encoders import TableEncoder
from .made import MadeCardinalityModel
from .training import TrainingSummary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def artifact_paths(output_dir: str | Path, schema_name: str, table_name: str) -> TableArtifacts:
    base = Path(output_dir).expanduser().resolve() / schema_name / table_name
    return TableArtifacts(
        table_name=table_name,
        schema_name=schema_name,
        table_dir=base,
        model_path=base / "model.pt",
        encoders_path=base / "encoders.json",
        config_path=base / "config.json",
        metadata_path=base / "metadata.json",
        training_summary_path=base / "training_summary.json",
    )


def save_artifacts(
    artifacts: TableArtifacts,
    model: MadeCardinalityModel,
    encoder: TableEncoder,
    config: NaruConfig,
    summary: TrainingSummary | None = None,
) -> None:
    artifacts.table_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "domain_sizes": encoder.domain_sizes,
            "embedding_dim": config.embedding_dim,
            "hidden_dims": config.hidden_dims,
        },
        artifacts.model_path,
    )
    _write_json(artifacts.encoders_path, encoder.to_dict())
    _write_json(artifacts.config_path, config.to_dict())
    _write_json(
        artifacts.metadata_path,
        {
            "artifact_version": config.artifact_version,
            "schema_name": encoder.schema_name,
            "table_name": encoder.table_name,
            "qualified_name": f"{encoder.schema_name}.{encoder.table_name}",
            "trained_columns": encoder.column_names,
            "config_snapshot": config.to_dict(),
            "row_count": encoder.row_count,
        },
    )
    if summary is not None and artifacts.training_summary_path is not None:
        _write_json(artifacts.training_summary_path, summary.to_dict())


def load_artifacts(artifacts: TableArtifacts) -> tuple[MadeCardinalityModel, TableEncoder, NaruConfig]:
    if not artifacts.is_complete():
        missing = [str(path.name) for path in artifacts.required_paths if not path.exists()]
        raise FileNotFoundError(
            f"Incomplete artifact set for {artifacts.schema_name}.{artifacts.table_name}: {', '.join(missing)}"
        )
    checkpoint = torch.load(artifacts.model_path, map_location="cpu")
    encoder_payload = json.loads(artifacts.encoders_path.read_text(encoding="utf-8"))
    config_payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    encoder = TableEncoder.from_dict(encoder_payload)
    config = NaruConfig.from_dict(config_payload)
    if metadata_payload.get("artifact_version") != config.artifact_version:
        raise ValueError(
            f"Artifact version mismatch for {metadata_payload.get('qualified_name', artifacts.table_dir.name)}"
        )
    if list(metadata_payload.get("trained_columns", [])) != list(encoder.column_names):
        raise ValueError(
            f"Artifact metadata mismatch for {metadata_payload.get('qualified_name', artifacts.table_dir.name)}"
        )
    model = MadeCardinalityModel(
        domain_sizes=list(checkpoint["domain_sizes"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        seed=config.random_seed,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, encoder, config

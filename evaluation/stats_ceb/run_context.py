from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from evaluation.stats_ceb.reports import write_json


EXPERIMENT_DIRS = {
    "experiment_1": "experiment_1_accuracy",
    "experiment_2": "experiment_2_failures",
    "experiment_3": "experiment_3_base_join_influence",
    "experiment_4": "experiment_4_base_join_quality",
    "experiment_5": "experiment_5_runtime",
}


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    output_root: Path
    run_dir: Path
    overwrite: bool
    resume: bool

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        run_id: str,
        *,
        overwrite: bool = False,
        resume: bool = False,
    ) -> "RunContext":
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty path-safe directory name")
        root = Path(output_root).expanduser().resolve()
        run_dir = root / run_id
        if run_dir.exists() and not (overwrite or resume):
            raise FileExistsError(
                f"Evaluation run already exists: {run_dir}. Use --resume or --overwrite explicitly."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("preflight", "plans", "sql", "plots"):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)
        return cls(run_id, root, run_dir, overwrite, resume)

    def experiment_dir(self, experiment: str) -> Path:
        relative = EXPERIMENT_DIRS[experiment]
        path = self.run_dir / relative
        if path.exists() and any(path.iterdir()) and not (self.overwrite or self.resume):
            raise FileExistsError(f"Experiment artifacts already exist: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.run_dir).as_posix()


def build_manifest(
    context: RunContext,
    *,
    database: Any,
    registry: Any,
    args: Any,
    workload_sources: Sequence[str | Path],
    workload_query_count: int,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    git_commit = _git(repository_root, "rev-parse", "HEAD")
    dirty = bool(_git(repository_root, "status", "--porcelain"))
    database_description = database.describe()
    described_counts = _table_counts(database_description)
    table_counts = _exact_table_counts(database, described_counts)
    model_artifacts: list[dict[str, Any]] = []
    estimator_configs: list[dict[str, Any]] = []
    for qualified_name, entry in sorted(registry.entries.items()):
        metadata = _read_json(entry.artifacts.metadata_path)
        config = _read_json(entry.artifacts.config_path)
        model_artifacts.append(
            {
                "table": qualified_name,
                "model_path": str(entry.artifacts.model_path),
                "metadata_path": str(entry.artifacts.metadata_path),
                "artifact_version": metadata.get("artifact_version"),
            }
        )
        estimator_configs.append(
            {
                "table": qualified_name,
                "sample_count": config.get("sample_count"),
                "random_seed": config.get("random_seed"),
                "max_enumeration_domain_product": config.get("max_enumeration_domain_product"),
            }
        )
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "available (name unavailable)"
    sanitized_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {"conn_string"}
    }
    return {
        "run_id": context.run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit or None,
        "dirty_worktree": dirty,
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "postbound_version": _package_version("postbound"),
        "postgresql_version": str(database.database_system_version()),
        "pg_lab_backend": _json_safe(database_description.get("hinting_mode")),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "database_name": database.database_name(),
        "database_schema": getattr(args, "schema", "public"),
        "database_table_row_counts": table_counts,
        "workload_sources": [str(Path(source).expanduser().resolve()) for source in workload_sources],
        "workload_query_count": workload_query_count,
        "model_artifacts": model_artifacts,
        "model_artifact_versions": sorted(
            {item["artifact_version"] for item in model_artifacts if item["artifact_version"] is not None}
        ),
        "estimator_configuration": estimator_configs,
        "sample_count_override": getattr(args, "sample_count", None),
        "random_seed": getattr(args, "random_seed", None),
        "bootstrap_seed": getattr(args, "bootstrap_seed", None),
        "bootstrap_samples": getattr(args, "bootstrap_samples", None),
        "timeout_seconds": getattr(args, "timeout_seconds", None),
        "warm_up_count": getattr(args, "warmups", None),
        "measured_repetition_count": getattr(args, "repetitions", None),
        "cache_policy": getattr(args, "cache_policy", None),
        "command_line_arguments": _json_safe(sanitized_args),
        "command_line": [sys.executable, *_redacted_argv(sys.argv)],
    }


def write_or_update_manifest(context: RunContext, manifest: dict[str, Any]) -> None:
    path = context.run_dir / "manifest.json"
    if path.exists() and context.resume:
        existing = _read_json(path)
        invocations = list(existing.get("invocations", []))
        invocations.append(
            {
                "timestamp_utc": manifest["timestamp_utc"],
                "command_line_arguments": manifest["command_line_arguments"],
                "command_line": manifest["command_line"],
            }
        )
        existing["invocations"] = invocations
        existing["workload_query_count"] = manifest["workload_query_count"]
        existing["workload_sources"] = sorted(
            set(existing.get("workload_sources", [])) | set(manifest["workload_sources"])
        )
        write_json(path, existing)
        return
    manifest["invocations"] = [
        {
            "timestamp_utc": manifest["timestamp_utc"],
            "command_line_arguments": manifest["command_line_arguments"],
            "command_line": manifest["command_line"],
        }
    ]
    write_json(path, manifest)


def _git(repository_root: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repository_root.as_posix()}",
        *args,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _table_counts(description: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in description.get("schema_info", []):
        if isinstance(payload, dict) and payload.get("table") is not None:
            counts[str(payload["table"])] = int(payload["n_rows"])
    return dict(sorted(counts.items()))


def _exact_table_counts(database: Any, described_counts: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in described_counts:
        safe_table = table.replace('"', '""')
        value = database.execute_query(f'SELECT COUNT(*) FROM "{safe_table}"', cache_enabled=False)
        counts[table] = int(value)
    return dict(sorted(counts.items()))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _redacted_argv(argv: list[str]) -> list[str]:
    output: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            output.append("<redacted>")
            redact_next = False
            continue
        output.append(value)
        redact_next = value == "--conn-string"
    return output

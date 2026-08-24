from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, payload: Any) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"CSV row contains non-finite value for {key}: {value!r}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_failures(output_dir: Path, failures: list[dict[str, Any]]) -> None:
    if not failures:
        return
    fields = ["label", "stage", "reason", "sql"]
    extra_fields = sorted({key for row in failures for key in row if key not in fields})
    write_csv(output_dir / "failures.csv", failures, fields + extra_fields)


def run_config(**kwargs: Any) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}

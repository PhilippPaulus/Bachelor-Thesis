from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from registry.registry import ModelRegistry


def test_registry_uses_schema_qualified_identity() -> None:
    base = Path.cwd() / ".tmp" / f"test-registry-{uuid.uuid4().hex}"
    try:
        base.mkdir(parents=True)
        for schema_name in ("s1", "s2"):
            table_dir = base / schema_name / "orders"
            table_dir.mkdir(parents=True)
            (table_dir / "model.pt").write_text("placeholder", encoding="utf-8")
            (table_dir / "encoders.json").write_text("{}", encoding="utf-8")
            (table_dir / "config.json").write_text("{}", encoding="utf-8")
            (table_dir / "metadata.json").write_text(
                json.dumps({"qualified_name": f"{schema_name}.orders"}),
                encoding="utf-8",
            )

        registry = ModelRegistry.load(base)

        assert registry.available_tables() == ["s1.orders", "s2.orders"]
        assert registry.has_model("s1.orders")
        assert not registry.has_model("orders")
    finally:
        shutil.rmtree(base, ignore_errors=True)

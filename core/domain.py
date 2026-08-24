from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from postbound import TableReference


class ColumnKind(str, Enum):
    INTEGER = "integer"
    NUMERIC = "numeric"
    FLOAT = "float"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    UNKNOWN = "unknown"


class PredicateOperator(str, Enum):
    EQ = "="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    BETWEEN = "between"
    IN = "in"


SUPPORTED_BACKEND_TYPES: dict[str, ColumnKind] = {
    "smallint": ColumnKind.INTEGER,
    "integer": ColumnKind.INTEGER,
    "bigint": ColumnKind.INTEGER,
    "numeric": ColumnKind.NUMERIC,
    "decimal": ColumnKind.NUMERIC,
    "real": ColumnKind.FLOAT,
    "double precision": ColumnKind.FLOAT,
    "text": ColumnKind.TEXT,
    "character varying": ColumnKind.TEXT,
    "character": ColumnKind.TEXT,
    "boolean": ColumnKind.BOOLEAN,
    "date": ColumnKind.DATE,
    "timestamp without time zone": ColumnKind.TIMESTAMP,
    "timestamp with time zone": ColumnKind.TIMESTAMP,
}

UNSUPPORTED_BACKEND_PREFIXES: tuple[str, ...] = (
    "ARRAY",
    "json",
    "jsonb",
    "USER-DEFINED",
)


@dataclass(frozen=True, slots=True)
class QualifiedTableName:
    schema_name: str
    table_name: str

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @classmethod
    def parse(
        cls,
        raw_name: str,
        *,
        default_schema: str | None = None,
    ) -> "QualifiedTableName":
        normalized = str(raw_name).strip()
        if "." in normalized:
            schema_name, table_name = normalized.split(".", 1)
            return cls(schema_name=schema_name, table_name=table_name)
        if default_schema is None:
            raise ValueError(f"Unqualified table name '{raw_name}' requires a schema")
        return cls(schema_name=default_schema, table_name=normalized)


@dataclass(slots=True)
class ColumnInfo:
    schema_name: str
    table_name: str
    column_name: str
    backend_type: str
    column_kind: ColumnKind
    ordinal_position: int
    is_nullable: bool
    supported: bool = True


@dataclass(slots=True)
class TableInfo:
    schema_name: str
    table_name: str
    estimated_row_count: int | None
    exact_row_count: int | None
    columns: list[ColumnInfo] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def supported_columns(self) -> list[ColumnInfo]:
        return [column for column in self.columns if column.supported]


@dataclass(slots=True)
class InternalPredicate:
    table_name: str
    column_name: str
    operator: PredicateOperator
    value: Any | None = None
    lower: Any | None = None
    upper: Any | None = None
    values: tuple[Any, ...] = ()


@dataclass(slots=True)
class TableArtifacts:
    table_name: str
    schema_name: str
    table_dir: Path
    model_path: Path
    encoders_path: Path
    config_path: Path
    metadata_path: Path
    training_summary_path: Path | None = None

    @property
    def required_paths(self) -> tuple[Path, ...]:
        return (
            self.model_path,
            self.encoders_path,
            self.config_path,
            self.metadata_path,
        )

    def is_complete(self) -> bool:
        return all(path.exists() for path in self.required_paths)


@dataclass(slots=True)
class EstimationResult:
    table_name: str
    selectivity: float
    cardinality: float
    used_model: bool
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationResult:
    table: TableReference | None
    predicates: list[InternalPredicate]
    fallback_reason: str | None = None

    @property
    def table_name(self) -> str | None:
        if self.table is None:
            return None
        return self.table.qualified_name() if self.table.schema else self.table.full_name

    @property
    def can_estimate(self) -> bool:
        return self.table is not None and self.fallback_reason is None


def ensure_tuple(values: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(values, tuple):
        return values
    if isinstance(values, (list, set, frozenset)):
        return tuple(values)
    return (values,)


def classify_backend_type(data_type: str) -> tuple[bool, ColumnKind]:
    normalized = data_type.strip().lower()
    if normalized in SUPPORTED_BACKEND_TYPES:
        return True, SUPPORTED_BACKEND_TYPES[normalized]
    if any(normalized.startswith(prefix.lower()) for prefix in UNSUPPORTED_BACKEND_PREFIXES):
        return False, ColumnKind.UNKNOWN
    return False, ColumnKind.UNKNOWN

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.config import NaruConfig
from core.domain import ColumnInfo, ColumnKind, InternalPredicate, PredicateOperator, TableInfo
from .discretization import NumericDiscretizer


_SENTINEL_NULL = "__NARU_NULL__"


class PredicateApplicationError(ValueError):
    pass


def _normalize_scalar(value: Any, column_kind: ColumnKind) -> Any:
    if pd.isna(value):
        return _SENTINEL_NULL
    if column_kind in {ColumnKind.INTEGER, ColumnKind.NUMERIC, ColumnKind.FLOAT}:
        return float(value)
    if column_kind == ColumnKind.BOOLEAN:
        return bool(value)
    if column_kind == ColumnKind.DATE:
        return pd.Timestamp(value).date().isoformat()
    if column_kind == ColumnKind.TIMESTAMP:
        return pd.Timestamp(value).isoformat()
    return str(value)


def _matches_range(value: Any, lower: Any | None, upper: Any | None) -> bool:
    if value == _SENTINEL_NULL:
        return False
    try:
        if lower is not None and value < lower:
            return False
        if upper is not None and value > upper:
            return False
    except TypeError:
        return False
    return True


@dataclass(slots=True)
class ColumnEncoder:
    column_name: str
    column_kind: ColumnKind
    use_discretizer: bool
    dictionary: dict[str, int]
    inverse_dictionary: list[Any]
    discretizer: NumericDiscretizer | None = None

    @classmethod
    def fit(cls, column: ColumnInfo, series: pd.Series, config: NaruConfig) -> "ColumnEncoder":
        normalized = series.map(lambda value: _normalize_scalar(value, column.column_kind))
        if column.column_kind in {ColumnKind.INTEGER, ColumnKind.NUMERIC, ColumnKind.FLOAT}:
            unique_count = normalized.nunique(dropna=False)
            if unique_count > config.numeric_max_unique_for_dictionary:
                discretizer = NumericDiscretizer.fit(series, config.numeric_bin_count)
                dictionary = {str(index): index for index in range(discretizer.domain_size)}
                inverse_dictionary = list(range(discretizer.domain_size))
                return cls(
                    column_name=column.column_name,
                    column_kind=column.column_kind,
                    use_discretizer=True,
                    dictionary=dictionary,
                    inverse_dictionary=inverse_dictionary,
                    discretizer=discretizer,
                )
        values = sorted(normalized.unique().tolist(), key=lambda value: (str(type(value)), value))
        dictionary = {json.dumps(value, sort_keys=True, default=str): idx for idx, value in enumerate(values)}
        return cls(
            column_name=column.column_name,
            column_kind=column.column_kind,
            use_discretizer=False,
            dictionary=dictionary,
            inverse_dictionary=values,
        )

    @property
    def domain_size(self) -> int:
        return len(self.inverse_dictionary)

    def encode_series(self, series: pd.Series) -> pd.Series:
        if self.use_discretizer:
            assert self.discretizer is not None
            return self.discretizer.transform_series(series)
        normalized = series.map(lambda value: _normalize_scalar(value, self.column_kind))
        return normalized.map(lambda value: self.dictionary[json.dumps(value, sort_keys=True, default=str)]).astype(int)

    def encode_scalar(self, value: Any) -> int:
        if self.use_discretizer:
            assert self.discretizer is not None
            return self.discretizer.transform_scalar(value)
        normalized = _normalize_scalar(value, self.column_kind)
        return self.dictionary[json.dumps(normalized, sort_keys=True, default=str)]

    def allowed_ids(self, predicate: InternalPredicate | None) -> list[int]:
        if predicate is None:
            return list(range(self.domain_size))
        if self.use_discretizer:
            assert self.discretizer is not None
            if predicate.operator == PredicateOperator.EQ:
                return [self.discretizer.transform_scalar(predicate.value)]
            if predicate.operator == PredicateOperator.LT:
                return self.discretizer.allowed_ids_for_range(upper=float(predicate.value), include_upper=False)
            if predicate.operator == PredicateOperator.LE:
                return self.discretizer.allowed_ids_for_range(upper=float(predicate.value), include_upper=True)
            if predicate.operator == PredicateOperator.GT:
                return self.discretizer.allowed_ids_for_range(lower=float(predicate.value), include_lower=False)
            if predicate.operator == PredicateOperator.GE:
                return self.discretizer.allowed_ids_for_range(lower=float(predicate.value), include_lower=True)
            if predicate.operator == PredicateOperator.BETWEEN:
                return self.discretizer.allowed_ids_for_range(
                    lower=float(predicate.lower),
                    upper=float(predicate.upper),
                    include_lower=True,
                    include_upper=True,
                )
            if predicate.operator == PredicateOperator.IN:
                return sorted({self.discretizer.transform_scalar(value) for value in predicate.values})
            raise ValueError(f"Unsupported predicate operator: {predicate.operator}")

        if predicate.operator == PredicateOperator.EQ:
            encoded = self._try_encode(predicate.value)
            return [] if encoded is None else [encoded]
        if predicate.operator == PredicateOperator.IN:
            values = {encoded for value in predicate.values if (encoded := self._try_encode(value)) is not None}
            return sorted(values)
        if predicate.operator in {PredicateOperator.LT, PredicateOperator.LE, PredicateOperator.GT, PredicateOperator.GE, PredicateOperator.BETWEEN}:
            ordered = self.inverse_dictionary
            if predicate.operator == PredicateOperator.LT:
                upper = _normalize_scalar(predicate.value, self.column_kind)
                return [
                    idx for idx, value in enumerate(ordered)
                    if value != _SENTINEL_NULL and _matches_range(value, None, upper) and value != upper
                ]
            if predicate.operator == PredicateOperator.LE:
                upper = _normalize_scalar(predicate.value, self.column_kind)
                return [idx for idx, value in enumerate(ordered) if _matches_range(value, None, upper)]
            if predicate.operator == PredicateOperator.GT:
                lower = _normalize_scalar(predicate.value, self.column_kind)
                return [
                    idx for idx, value in enumerate(ordered)
                    if value != _SENTINEL_NULL and _matches_range(value, lower, None) and value != lower
                ]
            if predicate.operator == PredicateOperator.GE:
                lower = _normalize_scalar(predicate.value, self.column_kind)
                return [idx for idx, value in enumerate(ordered) if _matches_range(value, lower, None)]
            assert predicate.operator == PredicateOperator.BETWEEN
            lower = _normalize_scalar(predicate.lower, self.column_kind)
            upper = _normalize_scalar(predicate.upper, self.column_kind)
            return [idx for idx, value in enumerate(ordered) if _matches_range(value, lower, upper)]
        raise ValueError(f"Unsupported predicate operator: {predicate.operator}")

    def _try_encode(self, value: Any) -> int | None:
        key = json.dumps(_normalize_scalar(value, self.column_kind), sort_keys=True, default=str)
        return self.dictionary.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "column_kind": self.column_kind.value,
            "use_discretizer": self.use_discretizer,
            "dictionary": self.dictionary,
            "inverse_dictionary": self.inverse_dictionary,
            "discretizer": None if self.discretizer is None else {"bin_edges": self.discretizer.bin_edges},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ColumnEncoder":
        discretizer_payload = payload.get("discretizer")
        discretizer = None if discretizer_payload is None else NumericDiscretizer(bin_edges=list(discretizer_payload["bin_edges"]))
        return cls(
            column_name=payload["column_name"],
            column_kind=ColumnKind(payload["column_kind"]),
            use_discretizer=bool(payload["use_discretizer"]),
            dictionary={str(key): int(value) for key, value in payload["dictionary"].items()},
            inverse_dictionary=list(payload["inverse_dictionary"]),
            discretizer=discretizer,
        )


@dataclass(slots=True)
class TableEncoder:
    schema_name: str
    table_name: str
    column_names: list[str]
    column_encoders: list[ColumnEncoder]
    row_count: int

    @classmethod
    def fit(
        cls,
        table_info: TableInfo,
        dataframe: pd.DataFrame,
        config: NaruConfig,
    ) -> tuple["TableEncoder", torch.Tensor]:
        columns = table_info.supported_columns
        if not columns:
            raise ValueError(f"Table {table_info.qualified_name} has no supported columns")
        encoded_columns: list[pd.Series] = []
        encoders: list[ColumnEncoder] = []
        for column in columns:
            encoder = ColumnEncoder.fit(column, dataframe[column.column_name], config)
            encoders.append(encoder)
            encoded_columns.append(encoder.encode_series(dataframe[column.column_name]))
        encoded_frame = pd.concat(encoded_columns, axis=1)
        tensor = torch.tensor(encoded_frame.to_numpy(dtype=np.int64), dtype=torch.long)
        row_count = int(table_info.exact_row_count or len(dataframe))
        table_encoder = cls(
            schema_name=table_info.schema_name,
            table_name=table_info.table_name,
            column_names=[column.column_name for column in columns],
            column_encoders=encoders,
            row_count=row_count,
        )
        return table_encoder, tensor

    def column_index(self, column_name: str) -> int:
        return self.column_names.index(column_name)

    def encoder_for(self, column_name: str) -> ColumnEncoder:
        return self.column_encoders[self.column_index(column_name)]

    @property
    def domain_sizes(self) -> list[int]:
        return [encoder.domain_size for encoder in self.column_encoders]

    @property
    def column_count(self) -> int:
        return len(self.column_encoders)

    def allowed_token_ids(self, predicates: list[InternalPredicate]) -> dict[int, list[int]]:
        predicates_by_column: dict[str, list[InternalPredicate]] = {}
        for predicate in predicates:
            predicates_by_column.setdefault(predicate.column_name, []).append(predicate)

        unknown_columns = sorted(set(predicates_by_column) - set(self.column_names))
        if unknown_columns:
            raise PredicateApplicationError(
                "Predicates reference columns that were not trained: " + ", ".join(unknown_columns)
            )

        allowed: dict[int, list[int]] = {}
        for index, encoder in enumerate(self.column_encoders):
            current_allowed = set(range(encoder.domain_size))
            for predicate in predicates_by_column.get(encoder.column_name, []):
                current_allowed &= set(encoder.allowed_ids(predicate))
            allowed[index] = sorted(current_allowed)
        return allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "table_name": self.table_name,
            "column_names": self.column_names,
            "row_count": self.row_count,
            "column_encoders": [encoder.to_dict() for encoder in self.column_encoders],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TableEncoder":
        return cls(
            schema_name=payload["schema_name"],
            table_name=payload["table_name"],
            column_names=list(payload["column_names"]),
            row_count=int(payload["row_count"]),
            column_encoders=[ColumnEncoder.from_dict(item) for item in payload["column_encoders"]],
        )

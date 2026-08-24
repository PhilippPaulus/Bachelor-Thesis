from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class NumericDiscretizer:
    bin_edges: list[float]

    @classmethod
    def fit(cls, series: pd.Series, bin_count: int) -> "NumericDiscretizer":
        numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if numeric.empty:
            return cls(bin_edges=[0.0, 1.0])
        quantiles = np.linspace(0.0, 1.0, num=min(bin_count, max(len(numeric.unique()), 2)))
        edges = np.unique(np.quantile(numeric.to_numpy(), quantiles))
        if len(edges) == 1:
            value = float(edges[0])
            edges = np.array([value, value + 1.0])
        return cls(bin_edges=[float(edge) for edge in edges])

    def transform_series(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        bins = pd.cut(
            numeric,
            bins=self.bin_edges,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )
        return bins.fillna(0).astype(int)

    def transform_scalar(self, value: object) -> int:
        if value is None:
            return 0
        scalar = float(value)
        bin_index = np.searchsorted(self.bin_edges, scalar, side="right") - 1
        return int(max(0, min(bin_index, len(self.bin_edges) - 2)))

    def allowed_ids_for_range(
        self,
        *,
        lower: float | None = None,
        upper: float | None = None,
        include_lower: bool = True,
        include_upper: bool = True,
    ) -> list[int]:
        allowed: list[int] = []
        for idx in range(len(self.bin_edges) - 1):
            left = self.bin_edges[idx]
            right = self.bin_edges[idx + 1]
            left_ok = True if lower is None else (right >= lower if include_lower else right > lower)
            right_ok = True if upper is None else (left <= upper if include_upper else left < upper)
            if left_ok and right_ok:
                allowed.append(idx)
        return allowed or [0]

    @property
    def domain_size(self) -> int:
        return max(1, len(self.bin_edges) - 1)

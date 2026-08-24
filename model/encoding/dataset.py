from __future__ import annotations

import torch
from torch.utils.data import Dataset


class EncodedTableDataset(Dataset[torch.Tensor]):
    def __init__(self, encoded_rows: torch.Tensor) -> None:
        if encoded_rows.ndim != 2:
            raise ValueError("encoded_rows must be a 2D tensor")
        self.encoded_rows = encoded_rows.long()

    def __len__(self) -> int:
        return int(self.encoded_rows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.encoded_rows[index]

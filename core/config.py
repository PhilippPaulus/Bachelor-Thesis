from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class NaruConfig:
    """Central configuration for training, persistence, inference, and artifacts."""

    schema_name: str = "public"
    batch_size: int = 1024
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    validation_split: float = 0.1
    hidden_dims: tuple[int, ...] = (128, 128, 128, 128)
    embedding_dim: int = 64
    numeric_max_unique_for_dictionary: int = 64
    numeric_bin_count: int = 64
    max_rows: int | None = None
    sampling_strategy: str = "limit"
    random_seed: int = 123
    device: str = "cuda"
    num_workers: int = 0
    sample_count: int = 8000
    max_enumeration_domain_product: int = 4096
    model_subdir: str = "models"
    save_training_summary: bool = True
    log_level: str = "INFO"
    use_rowcount_estimate: bool = False
    artifact_version: str = "v2"
    min_epochs_before_timeout: int = 10
    epoch_timeout_seconds_after_min_epochs: float | None = None
    table_allowlist: tuple[str, ...] = field(default_factory=tuple)
    table_denylist: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NaruConfig":
        return cls(**payload)

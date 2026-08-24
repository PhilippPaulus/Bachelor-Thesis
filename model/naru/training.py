from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split

from core.config import NaruConfig
from core.logging_utils import get_logger, log_kv
from ..encoding.dataset import EncodedTableDataset
from .made import MadeCardinalityModel

logger = get_logger(__name__)


def _render_progress(
    *,
    label: str,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    epoch_elapsed: float,
    train_loss: float | None = None,
    validation_loss: float | None = None,
    bar_width: int = 28,
    finished_epoch: bool = False,
) -> None:
    total_batches = max(total_batches, 1)
    ratio = min(max(batch / total_batches, 0.0), 1.0)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    metrics = []
    if train_loss is not None:
        metrics.append(f"train={train_loss:.4f}")
    if validation_loss is not None:
        metrics.append(f"val={validation_loss:.4f}")
    metric_text = " ".join(metrics)
    message = (
        f"\r{label} epoch {epoch}/{total_epochs} "
        f"[{bar}] {batch}/{total_batches} "
        f"elapsed={epoch_elapsed:7.1f}s {metric_text}"
    )
    print(message, end="\n" if finished_epoch else "", flush=True)


@dataclass(slots=True)
class TrainingSummary:
    train_loss_history: list[float]
    validation_loss_history: list[float]
    best_validation_loss: float
    epochs_completed: int
    epoch_durations_seconds: list[float]
    stopped_early: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "train_loss_history": self.train_loss_history,
            "validation_loss_history": self.validation_loss_history,
            "best_validation_loss": self.best_validation_loss,
            "epochs_completed": self.epochs_completed,
            "epoch_durations_seconds": self.epoch_durations_seconds,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


@torch.no_grad()
def _evaluate(model: MadeCardinalityModel, loader: DataLoader[torch.Tensor], device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        loss = model.negative_log_likelihood(batch)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train_model(
    encoded_rows: torch.Tensor,
    domain_sizes: list[int],
    config: NaruConfig,
    *,
    progress_label: str = "training",
) -> tuple[MadeCardinalityModel, TrainingSummary]:
    device = torch.device(config.device)
    dataset = EncodedTableDataset(encoded_rows)
    validation_size = max(1, int(len(dataset) * config.validation_split)) if len(dataset) > 1 else 0
    train_size = max(1, len(dataset) - validation_size)
    if validation_size > 0 and train_size + validation_size > len(dataset):
        validation_size = len(dataset) - train_size
    if validation_size > 0:
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(config.random_seed),
        )
    else:
        train_dataset = dataset
        validation_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = MadeCardinalityModel(
        domain_sizes=domain_sizes,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        seed=config.random_seed,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_history: list[float] = []
    validation_history: list[float] = []
    epoch_durations: list[float] = []
    best_validation = float("inf")
    best_state = None
    stopped_early = False
    stop_reason = None

    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        total_batches = len(train_loader)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.negative_log_likelihood(batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if batch_index == 1 or batch_index == total_batches or batch_index % max(1, total_batches // 20) == 0:
                _render_progress(
                    label=progress_label,
                    epoch=epoch + 1,
                    total_epochs=config.epochs,
                    batch=batch_index,
                    total_batches=total_batches,
                    epoch_elapsed=time.perf_counter() - epoch_started,
                    train_loss=float(sum(losses) / len(losses)),
                )
        train_loss = float(sum(losses) / max(len(losses), 1))
        _render_progress(
            label=progress_label,
            epoch=epoch + 1,
            total_epochs=config.epochs,
            batch=total_batches,
            total_batches=total_batches,
            epoch_elapsed=time.perf_counter() - epoch_started,
            train_loss=train_loss,
        )
        validation_loss = _evaluate(model, validation_loader, device)
        train_history.append(train_loss)
        validation_history.append(validation_loss)
        epoch_duration = time.perf_counter() - epoch_started
        epoch_durations.append(epoch_duration)
        _render_progress(
            label=progress_label,
            epoch=epoch + 1,
            total_epochs=config.epochs,
            batch=total_batches,
            total_batches=total_batches,
            epoch_elapsed=epoch_duration,
            train_loss=train_loss,
            validation_loss=validation_loss,
            finished_epoch=True,
        )
        log_kv(logger, "Epoch finished", epoch=epoch + 1, train_loss=train_loss, validation_loss=validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        completed_epochs = epoch + 1
        timeout_seconds = config.epoch_timeout_seconds_after_min_epochs
        if (
            timeout_seconds is not None
            and completed_epochs >= config.min_epochs_before_timeout
            and epoch_duration > timeout_seconds
        ):
            stopped_early = True
            stop_reason = (
                f"epoch {completed_epochs} took {epoch_duration:.2f}s, "
                f"exceeding timeout {timeout_seconds:.2f}s after at least "
                f"{config.min_epochs_before_timeout} epochs"
            )
            log_kv(logger, "Training stopped early", reason=stop_reason)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = TrainingSummary(
        train_loss_history=train_history,
        validation_loss_history=validation_history,
        best_validation_loss=best_validation,
        epochs_completed=len(train_history),
        epoch_durations_seconds=epoch_durations,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )
    return model, summary

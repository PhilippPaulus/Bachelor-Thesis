from __future__ import annotations

from typing import Sequence

import torch


def sample_from_probs(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> int:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def restrict_probabilities(
    probabilities: torch.Tensor,
    allowed_ids: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if len(allowed_ids) == 0:
        return torch.empty(0, device=probabilities.device), 0.0
    index = (
        allowed_ids.to(device=probabilities.device, dtype=torch.long)
        if isinstance(allowed_ids, torch.Tensor)
        else torch.tensor(list(allowed_ids), dtype=torch.long, device=probabilities.device)
    )
    restricted = probabilities.index_select(0, index)
    mass = float(restricted.sum().item())
    if mass <= 0.0:
        return restricted, 0.0
    return restricted / restricted.sum(), mass

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask.data.copy_(mask)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, self.weight * self.mask, self.bias)


@dataclass(slots=True)
class MadeMetadata:
    domain_sizes: list[int]
    embedding_dim: int
    hidden_dims: tuple[int, ...]


class MadeCardinalityModel(nn.Module):
    def __init__(
        self,
        domain_sizes: list[int],
        embedding_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not domain_sizes:
            raise ValueError("domain_sizes must not be empty")
        self.domain_sizes = domain_sizes
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.column_count = len(domain_sizes)
        self.total_output_dim = sum(domain_sizes)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(domain_size, embedding_dim) for domain_size in domain_sizes]
        )
        self.network = self._build_network(seed)

    @property
    def metadata(self) -> MadeMetadata:
        return MadeMetadata(
            domain_sizes=list(self.domain_sizes),
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
        )

    def _build_network(self, seed: int) -> nn.ModuleList:
        generator = torch.Generator()
        generator.manual_seed(seed)

        input_dim = self.column_count * self.embedding_dim
        layers = nn.ModuleList()
        dims = [input_dim, *self.hidden_dims, self.total_output_dim]
        masked_layers = [MaskedLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        hidden_degrees: list[torch.Tensor] = []
        input_degrees = torch.arange(1, self.column_count + 1).repeat_interleave(self.embedding_dim)
        previous_degrees = input_degrees

        for hidden_dim in self.hidden_dims:
            degrees = torch.randint(1, self.column_count, (hidden_dim,), generator=generator)
            hidden_degrees.append(degrees)
            layer = masked_layers[len(hidden_degrees) - 1]
            mask = (degrees[:, None] >= previous_degrees[None, :]).float()
            layer.set_mask(mask)
            layers.append(layer)
            layers.append(nn.ReLU())
            previous_degrees = degrees

        output_degrees = torch.arange(1, self.column_count + 1).repeat_interleave(torch.tensor(self.domain_sizes))
        output_layer = masked_layers[-1]
        output_mask = (output_degrees[:, None] > previous_degrees[None, :]).float()
        output_layer.set_mask(output_mask)
        layers.append(output_layer)
        return layers

    def forward(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("Expected shape [batch, columns]")
        pieces = [embedding(tokens[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        hidden = torch.cat(pieces, dim=-1)
        for layer in self.network:
            hidden = layer(hidden)
        logits = hidden
        return list(torch.split(logits, self.domain_sizes, dim=-1))

    def negative_log_likelihood(self, tokens: torch.Tensor) -> torch.Tensor:
        logits_per_column = self.forward(tokens)
        losses = []
        for idx, logits in enumerate(logits_per_column):
            losses.append(nn.functional.cross_entropy(logits, tokens[:, idx], reduction="none"))
        return torch.stack(losses, dim=1).sum(dim=1).mean()

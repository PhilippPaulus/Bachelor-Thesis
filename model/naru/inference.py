from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from core.config import NaruConfig
from core.domain import EstimationResult, InternalPredicate
from ..encoding.encoders import TableEncoder
from .made import MadeCardinalityModel
from .sampling import restrict_probabilities, sample_from_probs


@dataclass(slots=True)
class NaruTableEstimator:
    model: MadeCardinalityModel
    encoder: TableEncoder
    config: NaruConfig
    device: torch.device
    estimation_seed: int
    sample_count: int

    @classmethod
    def create(
        cls,
        model: MadeCardinalityModel,
        encoder: TableEncoder,
        config: NaruConfig,
        *,
        estimation_seed: int | None = None,
        sample_count: int | None = None,
    ) -> "NaruTableEstimator":
        device = torch.device(config.device)
        model = model.to(device)
        model.eval()
        return cls(
            model=model,
            encoder=encoder,
            config=config,
            device=device,
            estimation_seed=config.random_seed if estimation_seed is None else int(estimation_seed),
            sample_count=config.sample_count if sample_count is None else int(sample_count),
        )

    @torch.no_grad()
    def estimate(self, predicates: list[InternalPredicate]) -> EstimationResult:
        if not predicates:
            return EstimationResult(
                table_name=self.encoder.table_name,
                selectivity=1.0,
                cardinality=float(self.encoder.row_count),
                used_model=True,
                diagnostics={
                    "inference_mode": "full_table",
                    "sample_count": 0,
                    "estimator_seed": self.estimation_seed,
                    "constrained_columns": {},
                },
            )
        allowed = self.encoder.allowed_token_ids(predicates)
        enumerate_product = 1
        for token_ids in allowed.values():
            enumerate_product *= max(len(token_ids), 1)
        if enumerate_product <= self.config.max_enumeration_domain_product:
            selectivity = self._enumerate_exact(allowed)
            return EstimationResult(
                table_name=self.encoder.table_name,
                selectivity=selectivity,
                cardinality=selectivity * self.encoder.row_count,
                used_model=True,
                diagnostics=self._diagnostics(
                    predicates,
                    allowed,
                    inference_mode="enumeration",
                    accepted_masses=None,
                ),
            )
        selectivity, accepted_masses = self._progressive_sampling(allowed)
        return EstimationResult(
            table_name=self.encoder.table_name,
            selectivity=selectivity,
            cardinality=selectivity * self.encoder.row_count,
            used_model=True,
            diagnostics=self._diagnostics(
                predicates,
                allowed,
                inference_mode="progressive_sampling",
                accepted_masses=accepted_masses,
            ),
        )

    def configure_evaluation(self, *, estimation_seed: int, sample_count: int | None = None) -> None:
        self.estimation_seed = int(estimation_seed)
        if sample_count is not None:
            if sample_count <= 0:
                raise ValueError("sample_count must be positive")
            self.sample_count = int(sample_count)

    @torch.no_grad()
    def _enumerate_exact(self, allowed: dict[int, list[int]]) -> float:
        rows: list[list[int]] = [[]]
        for index in range(self.encoder.column_count):
            next_rows: list[list[int]] = []
            for row in rows:
                for token in allowed[index]:
                    next_rows.append([*row, token])
            rows = next_rows
        if not rows:
            return 0.0
        tokens = torch.tensor(rows, dtype=torch.long, device=self.device)
        return float(self._joint_probabilities(tokens).sum().item())

    @torch.no_grad()
    def _joint_probabilities(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.model(tokens)
        probabilities = []
        for index, column_logits in enumerate(logits):
            probs = torch.softmax(column_logits, dim=-1)
            probability = probs.gather(1, tokens[:, index:index + 1]).squeeze(1)
            probabilities.append(probability)
        return torch.stack(probabilities, dim=1).prod(dim=1)

    @torch.no_grad()
    def _progressive_sampling(self, allowed: dict[int, list[int]]) -> tuple[float, dict[int, float]]:
        draws: list[float] = []
        masses_by_column: dict[int, list[float]] = {index: [] for index in allowed}
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.estimation_seed)
        allowed_tensors = {
            index: torch.tensor(token_ids, dtype=torch.long, device=self.device)
            for index, token_ids in allowed.items()
        }
        for _ in range(self.sample_count):
            draw, masses = self._draw_one(
                allowed,
                allowed_tensors=allowed_tensors,
                generator=generator,
            )
            draws.append(draw)
            for index, mass in masses.items():
                masses_by_column[index].append(mass)
        mean_masses = {
            index: float(sum(values) / len(values))
            for index, values in masses_by_column.items()
            if values
        }
        return float(sum(draws) / max(len(draws), 1)), mean_masses

    @torch.no_grad()
    def _draw_one(
        self,
        allowed: dict[int, list[int]],
        *,
        allowed_tensors: dict[int, torch.Tensor],
        generator: torch.Generator,
    ) -> tuple[float, dict[int, float]]:
        token_buffer = torch.zeros((1, self.encoder.column_count), dtype=torch.long, device=self.device)
        estimate = 1.0
        masses: dict[int, float] = {}
        for index in range(self.encoder.column_count):
            logits = self.model(token_buffer)[index].squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            restricted, mass = restrict_probabilities(probs, allowed_tensors[index])
            masses[index] = mass
            if mass <= 0.0:
                return 0.0, masses
            estimate *= mass
            sampled_idx = sample_from_probs(restricted, generator=generator)
            token_buffer[0, index] = int(allowed[index][sampled_idx])
        return estimate, masses

    def _diagnostics(
        self,
        predicates: list[InternalPredicate],
        allowed: dict[int, list[int]],
        *,
        inference_mode: str,
        accepted_masses: dict[int, float] | None,
    ) -> dict[str, object]:
        constrained = {predicate.column_name for predicate in predicates}
        columns: dict[str, dict[str, object]] = {}
        for index, encoder in enumerate(self.encoder.column_encoders):
            if encoder.column_name not in constrained:
                continue
            token_ids = list(allowed[index])
            if len(token_ids) <= 256:
                token_payload: dict[str, object] = {"allowed_token_ids": token_ids}
            else:
                digest = hashlib.sha256(
                    ",".join(str(token) for token in token_ids).encode("ascii")
                ).hexdigest()
                token_payload = {
                    "allowed_token_ids": None,
                    "allowed_token_ids_compact": {
                        "count": len(token_ids),
                        "minimum": min(token_ids),
                        "maximum": max(token_ids),
                        "sha256": digest,
                    },
                }
            columns[encoder.column_name] = {
                "encoding_type": "numeric_discretizer" if encoder.use_discretizer else "dictionary",
                **token_payload,
                "allowed_token_count": len(token_ids),
                "domain_size": encoder.domain_size,
                "accepted_probability_mass": None
                if accepted_masses is None
                else accepted_masses.get(index),
            }
        return {
            "inference_mode": inference_mode,
            "sample_count": self.sample_count if inference_mode == "progressive_sampling" else 0,
            "estimator_seed": self.estimation_seed,
            "constrained_columns": columns,
        }

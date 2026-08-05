from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class TrajectoryOutput(NamedTuple):
    context: Tensor
    memory: Tensor
    sequence: Tensor


class ContinuousCycleEncoding(nn.Module):
    def __init__(self, dimension: int, maximum_cycles: int) -> None:
        super().__init__()
        self.cycle_embedding = nn.Embedding(maximum_cycles, dimension)
        self.elapsed_projection = nn.Sequential(
            nn.Linear(4, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, elapsed_hours: Tensor) -> Tensor:
        batch, cycles = elapsed_hours.shape
        indices = torch.arange(cycles, device=elapsed_hours.device).clamp_max(
            self.cycle_embedding.num_embeddings - 1
        )
        cycle = self.cycle_embedding(indices).unsqueeze(0).expand(batch, -1, -1)
        scaled = elapsed_hours / (24.0 * 7.0)
        features = torch.stack(
            (
                scaled,
                torch.log1p(scaled.clamp_min(0.0)),
                torch.sin(scaled * math.pi),
                torch.cos(scaled * math.pi),
            ),
            dim=-1,
        )
        return cycle + self.elapsed_projection(features)


class CausalTrajectoryReasoner(nn.Module):
    def __init__(
        self,
        dimension: int,
        memory_dimension: int,
        heads: int,
        layers: int,
        maximum_cycles: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position = ContinuousCycleEncoding(dimension, maximum_cycles)
        self.memory_to_state = nn.Linear(memory_dimension, dimension)
        block = nn.TransformerEncoderLayer(
            dimension,
            heads,
            dimension * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.memory_cell = nn.GRUCell(dimension, memory_dimension)
        self.context_projection = nn.Linear(dimension + memory_dimension, dimension)
        self.normalization = nn.LayerNorm(dimension)

    def causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        states: Tensor,
        elapsed_hours: Tensor,
        previous_memory: Tensor | None = None,
        valid_cycles: Tensor | None = None,
    ) -> TrajectoryOutput:
        batch, cycles, _ = states.shape
        if previous_memory is None:
            previous_memory = states.new_zeros(batch, self.memory_cell.hidden_size)
        memory_bias = self.memory_to_state(previous_memory).unsqueeze(1)
        inputs = states + self.position(elapsed_hours) + memory_bias
        padding_mask = None if valid_cycles is None else ~valid_cycles
        sequence = self.transformer(
            inputs,
            mask=self.causal_mask(cycles, states.device),
            src_key_padding_mask=padding_mask,
        )
        if valid_cycles is None:
            final = sequence[:, -1]
        else:
            positions = valid_cycles.long().sum(dim=1).sub(1).clamp_min(0)
            final = sequence[torch.arange(batch, device=states.device), positions]
        memory = self.memory_cell(final, previous_memory)
        context = self.context_projection(torch.cat((final, memory), dim=-1))
        return TrajectoryOutput(self.normalization(context), memory, sequence)


class IncrementalTrajectoryState(nn.Module):
    def __init__(self, reasoner: CausalTrajectoryReasoner, maximum_cycles: int) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.maximum_cycles = maximum_cycles

    def forward(
        self,
        state_history: list[Tensor],
        elapsed_history: list[Tensor],
        memory: Tensor | None,
    ) -> TrajectoryOutput:
        if len(state_history) == 0:
            raise ValueError("state history cannot be empty")
        if len(state_history) != len(elapsed_history):
            raise ValueError("state and elapsed histories must have equal length")
        states = torch.stack(state_history[-self.maximum_cycles :], dim=1)
        elapsed = torch.stack(elapsed_history[-self.maximum_cycles :], dim=1)
        return self.reasoner(states, elapsed, memory)

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tumor_orchestration.encoders import (
    CTVolumeEncoder,
    GenomicPathwayEncoder,
    PharmacologyGraphEncoder,
    StructuredEHREncoder,
)
from tumor_orchestration.fusion import CTIGA, FusionOutput, WeightedTokenPool
from tumor_orchestration.policy import PolicyOutput, SafeOrchestrator
from tumor_orchestration.records import AgentPredictions, CycleBatch, ModalityTokens
from tumor_orchestration.schema import AgentConfig
from tumor_orchestration.trajectory import CausalTrajectoryReasoner, TrajectoryOutput


class CycleEncoding(NamedTuple):
    state: Tensor
    fusion: FusionOutput


class MulticycleOutput(NamedTuple):
    predictions: AgentPredictions
    tokens: ModalityTokens
    trajectory: TrajectoryOutput


class SegmentationDecoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        widths = (dimension, dimension // 2, dimension // 4, dimension // 8)
        self.seed = nn.Linear(dimension, widths[0] * 2 * 2 * 2)
        layers: list[nn.Module] = []
        for input_width, output_width in zip(widths[:-1], widths[1:], strict=True):
            layers.extend(
                (
                    nn.ConvTranspose3d(
                        input_width,
                        output_width,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(1, output_width),
                    nn.GELU(),
                )
            )
        self.layers = nn.Sequential(*layers)
        self.output = nn.Conv3d(widths[-1], 1, kernel_size=1)

    def forward(self, state: Tensor, output_shape: tuple[int, int, int]) -> Tensor:
        hidden = self.seed(state).reshape(state.shape[0], -1, 2, 2, 2)
        logits = self.output(self.layers(hidden))
        return F.interpolate(logits, size=output_shape, mode="trilinear", align_corners=False)


class PredictionHeads(nn.Module):
    def __init__(self, dimension: int, toxicity_labels: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.response = nn.Linear(dimension, 1)
        self.survival = nn.Linear(dimension, 1)
        self.toxicity = nn.Linear(dimension, toxicity_labels)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.shared(context)
        return (
            self.response(hidden).squeeze(-1),
            self.survival(hidden).squeeze(-1),
            self.toxicity(hidden),
        )


class ClosedLoopTumorAgent(nn.Module):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.model.dimension
        self.ct_encoder = CTVolumeEncoder(dimension, config.data.image_tokens)
        self.genomic_encoder = GenomicPathwayEncoder(config.data, config.model)
        self.ehr_encoder = StructuredEHREncoder(
            config.data.ehr_features,
            dimension,
            config.data.ehr_tokens,
            config.model.dropout,
        )
        self.pharmacology_encoder = PharmacologyGraphEncoder(config.data, config.model)
        self.treatment_embedding = nn.Embedding(config.model.actions + 1, dimension)
        self.ctiga = CTIGA(
            dimension,
            config.model.heads,
            config.model.ctiga_layers,
            config.model.dropout,
        )
        self.pool = WeightedTokenPool(dimension)
        self.trajectory = CausalTrajectoryReasoner(
            dimension,
            config.model.memory_dimension,
            config.model.heads,
            config.model.temporal_layers,
            config.data.maximum_cycles,
            config.model.dropout,
        )
        self.heads = PredictionHeads(
            dimension,
            config.model.toxicity_labels,
            config.model.dropout,
        )
        self.segmentation = SegmentationDecoder(dimension)
        self.orchestrator = SafeOrchestrator(
            dimension,
            config.model.actions,
            config.model.dropout,
            config.safety.toxicity_threshold,
            config.safety.monte_carlo_samples,
        )

    def encode_modalities(self, cycle: CycleBatch) -> ModalityTokens:
        imaging = self.ct_encoder(cycle.ct)
        genomics = self.genomic_encoder(cycle.expression, cycle.mutations)
        ehr = self.ehr_encoder(cycle.ehr, cycle.ehr_times)
        pharmacology = self.pharmacology_encoder(cycle.drug_nodes, cycle.drug_adjacency)
        return ModalityTokens(imaging, genomics, ehr, pharmacology)

    def encode_cycle(self, cycle: CycleBatch) -> CycleEncoding:
        tokens = self.encode_modalities(cycle)
        treatment = self.treatment_embedding(cycle.treatment.long())
        fusion = self.ctiga(tokens, treatment, cycle.modality_mask.float())
        return CycleEncoding(self.pool(fusion.fused), fusion)

    def forward(
        self,
        cycles: tuple[CycleBatch, ...],
        elapsed_hours: Tensor,
        valid_cycles: Tensor | None = None,
        previous_memory: Tensor | None = None,
    ) -> MulticycleOutput:
        if len(cycles) == 0:
            raise ValueError("at least one cycle is required")
        encoded = [self.encode_cycle(cycle) for cycle in cycles]
        states = torch.stack([item.state for item in encoded], dim=1)
        trajectory = self.trajectory(states, elapsed_hours, previous_memory, valid_cycles)
        response, survival, toxicity = self.heads(trajectory.context)
        policy: PolicyOutput = self.orchestrator(trajectory.context)
        output_shape = tuple(int(item) for item in cycles[-1].ct.shape[-3:])
        segmentation = self.segmentation(encoded[-1].state, output_shape)
        predictions = AgentPredictions(
            response,
            survival,
            toxicity,
            segmentation,
            policy.logits,
            trajectory.context,
            trajectory.memory,
            torch.stack([item.fusion.gates for item in encoded], dim=1),
        )
        return MulticycleOutput(predictions, encoded[-1].fusion.modalities, trajectory)

    def parameter_groups(self) -> tuple[dict[str, object], ...]:
        encoders: list[nn.Parameter] = []
        for module in (
            self.ct_encoder,
            self.genomic_encoder,
            self.ehr_encoder,
            self.pharmacology_encoder,
        ):
            encoders.extend(module.parameters())
        encoder_ids = {id(parameter) for parameter in encoders}
        remaining = [
            parameter for parameter in self.parameters() if id(parameter) not in encoder_ids
        ]
        return (
            {"params": encoders, "name": "encoders"},
            {"params": remaining, "name": "agent"},
        )

    def freeze_encoders(self, frozen: bool = True) -> None:
        for module in (
            self.ct_encoder,
            self.genomic_encoder,
            self.ehr_encoder,
            self.pharmacology_encoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)

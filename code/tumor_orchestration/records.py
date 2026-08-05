from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor


class ModalityTokens(NamedTuple):
    imaging: Tensor
    genomics: Tensor
    ehr: Tensor
    pharmacology: Tensor


class AgentPredictions(NamedTuple):
    response_logits: Tensor
    survival_risk: Tensor
    toxicity_logits: Tensor
    segmentation_logits: Tensor
    policy_logits: Tensor
    state: Tensor
    memory: Tensor
    gates: Tensor


class TargetBatch(NamedTuple):
    response: Tensor
    survival_time: Tensor
    survival_event: Tensor
    toxicity: Tensor
    segmentation: Tensor
    actions: Tensor
    advantages: Tensor
    old_log_probabilities: Tensor


class LossBreakdown(NamedTuple):
    total: Tensor
    response: Tensor
    survival: Tensor
    toxicity: Tensor
    segmentation: Tensor
    policy: Tensor
    alignment: Tensor


class SafetyDecision(NamedTuple):
    action: Tensor
    uncertainty: Tensor
    escalated: Tensor
    safe_mask: Tensor
    probabilities: Tensor


class CycleBatch(NamedTuple):
    ct: Tensor
    expression: Tensor
    mutations: Tensor
    ehr: Tensor
    ehr_times: Tensor
    drug_nodes: Tensor
    drug_adjacency: Tensor
    treatment: Tensor
    modality_mask: Tensor


@dataclass(frozen=True)
class PatientTrajectory:
    patient_id: str
    cycles: tuple[CycleBatch, ...]
    targets: TargetBatch

    def __len__(self) -> int:
        return len(self.cycles)

    def __iter__(self) -> Iterator[CycleBatch]:
        return iter(self.cycles)


@dataclass(frozen=True)
class DrugConstraintProfile:
    contraindicated: Tensor
    interaction_risk: Tensor
    allergy_risk: Tensor
    comorbidity_risk: Tensor

    def combined_mask(self, toxicity_probability: Tensor, threshold: float) -> Tensor:
        structural = self.contraindicated.bool()
        interactions = self.interaction_risk >= threshold
        allergies = self.allergy_risk >= threshold
        comorbidities = self.comorbidity_risk >= threshold
        toxicity = toxicity_probability >= threshold
        return ~(structural | interactions | allergies | comorbidities | toxicity)


@dataclass(frozen=True)
class CohortSplit:
    training: tuple[str, ...]
    validation: tuple[str, ...]
    testing: tuple[str, ...]

    def validate(self) -> None:
        training = set(self.training)
        validation = set(self.validation)
        testing = set(self.testing)
        if training & validation or training & testing or validation & testing:
            raise ValueError("patient identifiers overlap between partitions")


@dataclass(frozen=True)
class RunningMoments:
    count: int
    mean: Tensor
    second: Tensor

    @classmethod
    def initialize(cls, shape: tuple[int, ...], device: torch.device) -> RunningMoments:
        return cls(0, torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    def update(self, values: Tensor) -> RunningMoments:
        flat = values.reshape(-1, *self.mean.shape)
        count = flat.shape[0]
        if count == 0:
            return self
        batch_mean = flat.mean(dim=0)
        batch_second = ((flat - batch_mean) ** 2).sum(dim=0)
        total = self.count + count
        delta = batch_mean - self.mean
        mean = self.mean + delta * count / max(total, 1)
        second = self.second + batch_second + delta.square() * self.count * count / max(total, 1)
        return RunningMoments(total, mean, second)

    def variance(self) -> Tensor:
        return self.second / max(self.count - 1, 1)


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    phase: str
    loss: float
    response_auc: float
    survival_c_index: float
    toxicity_auc: float
    dice: float
    learning_rate: float

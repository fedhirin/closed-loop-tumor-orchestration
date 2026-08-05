from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class DataConfig:
    ct_patch: tuple[int, int, int]
    image_tokens: int
    gene_expression_features: int
    mutation_features: int
    genomic_tokens: int
    ehr_features: int
    ehr_tokens: int
    drug_fingerprint_features: int
    pharmacology_tokens: int
    maximum_cycles: int


@dataclass(frozen=True)
class ModelConfig:
    dimension: int
    heads: int
    ctiga_layers: int
    genomic_layers: int
    pharmacology_layers: int
    temporal_layers: int
    memory_dimension: int
    actions: int
    toxicity_labels: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    supervised_epochs: int
    policy_episodes: int
    batch_size_per_device: int
    gradient_accumulation: int
    world_size: int
    effective_batch_size: int
    learning_rate: float
    policy_learning_rate: float
    weight_decay: float
    warmup_epochs: int
    early_stopping_patience: int
    precision: str
    scheduler: str
    optimizer: str
    ppo_clip: float
    seeds: int


@dataclass(frozen=True)
class LossConfig:
    survival: float
    toxicity: float
    segmentation: float
    policy: float
    alignment: float


@dataclass(frozen=True)
class SafetyConfig:
    efficacy: float
    toxicity: float
    violation_penalty: float
    toxicity_threshold: float
    escalation_percentile: float
    monte_carlo_samples: int


@dataclass(frozen=True)
class EvaluationConfig:
    folds: int
    validation_fraction: float
    bootstrap_iterations: int
    bonferroni_comparisons: int


@dataclass(frozen=True)
class AgentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    safety: SafetyConfig
    evaluation: EvaluationConfig

    def validate(self) -> None:
        if self.model.dimension % self.model.heads != 0:
            raise ValueError("dimension must be divisible by heads")
        effective = (
            self.training.batch_size_per_device
            * self.training.gradient_accumulation
            * self.training.world_size
        )
        if effective != self.training.effective_batch_size:
            raise ValueError("effective batch size is inconsistent")
        if not 0.0 < self.safety.toxicity_threshold < 1.0:
            raise ValueError("toxicity threshold must be between zero and one")
        if self.data.maximum_cycles < 1:
            raise ValueError("maximum cycles must be positive")
        if len(self.data.ct_patch) != 3:
            raise ValueError("ct patch must contain three dimensions")


def _mapping(value: object, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return cast(Mapping[str, Any], value)


def _integer(source: Mapping[str, Any], key: str) -> int:
    value = source[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _float(source: Mapping[str, Any], key: str) -> float:
    value = source[key]
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _string(source: Mapping[str, Any], key: str) -> str:
    value = source[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be text")
    return value


def _triple(source: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    value = source[key]
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise TypeError(f"{key} must contain three integers")
    result = tuple(int(item) for item in value)
    return cast(tuple[int, int, int], result)


def parse_config(raw: Mapping[str, Any]) -> AgentConfig:
    data = _mapping(raw["data"], "data")
    model = _mapping(raw["model"], "model")
    training = _mapping(raw["training"], "training")
    loss = _mapping(raw["loss"], "loss")
    safety = _mapping(raw["safety"], "safety")
    evaluation = _mapping(raw["evaluation"], "evaluation")
    config = AgentConfig(
        seed=_integer(raw, "seed"),
        data=DataConfig(
            ct_patch=_triple(data, "ct_patch"),
            image_tokens=_integer(data, "image_tokens"),
            gene_expression_features=_integer(data, "gene_expression_features"),
            mutation_features=_integer(data, "mutation_features"),
            genomic_tokens=_integer(data, "genomic_tokens"),
            ehr_features=_integer(data, "ehr_features"),
            ehr_tokens=_integer(data, "ehr_tokens"),
            drug_fingerprint_features=_integer(data, "drug_fingerprint_features"),
            pharmacology_tokens=_integer(data, "pharmacology_tokens"),
            maximum_cycles=_integer(data, "maximum_cycles"),
        ),
        model=ModelConfig(
            dimension=_integer(model, "dimension"),
            heads=_integer(model, "heads"),
            ctiga_layers=_integer(model, "ctiga_layers"),
            genomic_layers=_integer(model, "genomic_layers"),
            pharmacology_layers=_integer(model, "pharmacology_layers"),
            temporal_layers=_integer(model, "temporal_layers"),
            memory_dimension=_integer(model, "memory_dimension"),
            actions=_integer(model, "actions"),
            toxicity_labels=_integer(model, "toxicity_labels"),
            dropout=_float(model, "dropout"),
        ),
        training=TrainingConfig(
            supervised_epochs=_integer(training, "supervised_epochs"),
            policy_episodes=_integer(training, "policy_episodes"),
            batch_size_per_device=_integer(training, "batch_size_per_device"),
            gradient_accumulation=_integer(training, "gradient_accumulation"),
            world_size=_integer(training, "world_size"),
            effective_batch_size=_integer(training, "effective_batch_size"),
            learning_rate=_float(training, "learning_rate"),
            policy_learning_rate=_float(training, "policy_learning_rate"),
            weight_decay=_float(training, "weight_decay"),
            warmup_epochs=_integer(training, "warmup_epochs"),
            early_stopping_patience=_integer(training, "early_stopping_patience"),
            precision=_string(training, "precision"),
            scheduler=_string(training, "scheduler"),
            optimizer=_string(training, "optimizer"),
            ppo_clip=_float(training, "ppo_clip"),
            seeds=_integer(training, "seeds"),
        ),
        loss=LossConfig(
            survival=_float(loss, "survival"),
            toxicity=_float(loss, "toxicity"),
            segmentation=_float(loss, "segmentation"),
            policy=_float(loss, "policy"),
            alignment=_float(loss, "alignment"),
        ),
        safety=SafetyConfig(
            efficacy=_float(safety, "efficacy"),
            toxicity=_float(safety, "toxicity"),
            violation_penalty=_float(safety, "violation_penalty"),
            toxicity_threshold=_float(safety, "toxicity_threshold"),
            escalation_percentile=_float(safety, "escalation_percentile"),
            monte_carlo_samples=_integer(safety, "monte_carlo_samples"),
        ),
        evaluation=EvaluationConfig(
            folds=_integer(evaluation, "folds"),
            validation_fraction=_float(evaluation, "validation_fraction"),
            bootstrap_iterations=_integer(evaluation, "bootstrap_iterations"),
            bonferroni_comparisons=_integer(evaluation, "bonferroni_comparisons"),
        ),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> AgentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return parse_config(_mapping(raw, "root"))

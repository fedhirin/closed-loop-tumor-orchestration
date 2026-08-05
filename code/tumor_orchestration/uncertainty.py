from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn


class UncertaintySummary(NamedTuple):
    predictive_entropy: Tensor
    expected_entropy: Tensor
    mutual_information: Tensor
    variation_ratio: Tensor
    mean_probability: Tensor


@dataclass(frozen=True)
class EscalationCalibration:
    percentile: float
    threshold: float
    validation_rate: float
    retained_accuracy: float


def entropy(probabilities: Tensor) -> Tensor:
    bounded = probabilities.clamp_min(torch.finfo(probabilities.dtype).eps)
    return -(bounded * bounded.log()).sum(dim=-1)


def summarize_samples(probabilities: Tensor) -> UncertaintySummary:
    if probabilities.ndim != 3:
        raise ValueError("probability samples require sample, batch, and class dimensions")
    mean_probability = probabilities.mean(dim=0)
    predictive = entropy(mean_probability)
    expected = entropy(probabilities).mean(dim=0)
    mutual_information = predictive - expected
    classes = probabilities.argmax(dim=-1)
    mode_counts = []
    for patient in range(classes.shape[1]):
        counts = torch.bincount(classes[:, patient], minlength=probabilities.shape[-1])
        mode_counts.append(counts.max())
    maximum = torch.stack(mode_counts).float()
    variation = 1.0 - maximum / probabilities.shape[0]
    return UncertaintySummary(predictive, expected, mutual_information, variation, mean_probability)


def dropout_probabilities(
    module: nn.Module,
    inputs: Tensor,
    forward: Callable[[nn.Module, Tensor], Tensor],
    samples: int,
) -> Tensor:
    previous = module.training
    module.train(True)
    outputs = []
    for _ in range(samples):
        outputs.append(torch.softmax(forward(module, inputs), dim=-1))
    module.train(previous)
    return torch.stack(outputs)


def calibrate_escalation(
    uncertainty: Tensor,
    correct: Tensor,
    percentile: float,
) -> EscalationCalibration:
    threshold = torch.quantile(uncertainty.float(), percentile / 100.0)
    retained = uncertainty <= threshold
    rate = float((~retained).float().mean().item())
    accuracy = float(correct[retained].float().mean().item()) if retained.any() else float("nan")
    return EscalationCalibration(percentile, float(threshold.item()), rate, accuracy)


def risk_coverage_curve(
    uncertainty: Tensor,
    correct: Tensor,
    coverages: Sequence[float] | None = None,
) -> tuple[tuple[float, float, float], ...]:
    requested = coverages or tuple(np.linspace(0.1, 1.0, 10).tolist())
    order = torch.argsort(uncertainty)
    sorted_correct = correct[order].float()
    sorted_uncertainty = uncertainty[order]
    points: list[tuple[float, float, float]] = []
    for coverage in requested:
        count = max(1, round(len(order) * coverage))
        risk = 1.0 - float(sorted_correct[:count].mean().item())
        threshold = float(sorted_uncertainty[count - 1].item())
        points.append((float(coverage), risk, threshold))
    return tuple(points)


def area_under_risk_coverage(points: Sequence[tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    coverages = np.asarray([point[0] for point in points], dtype=float)
    risks = np.asarray([point[1] for point in points], dtype=float)
    return float(np.trapz(risks, coverages))


def expected_calibration_error_multiclass(
    probabilities: Tensor,
    targets: Tensor,
    bins: int = 15,
) -> Tensor:
    confidence, predictions = probabilities.max(dim=-1)
    correctness = predictions.eq(targets).float()
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    result = probabilities.new_zeros(())
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= boundaries[index]) & (confidence <= boundaries[index + 1])
        else:
            selected = (confidence >= boundaries[index]) & (confidence < boundaries[index + 1])
        if selected.any():
            difference = (correctness[selected].mean() - confidence[selected].mean()).abs()
            result = result + difference * selected.float().mean()
    return result


def temperature_scale(logits: Tensor, targets: Tensor, iterations: int = 100) -> Tensor:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=iterations)
    criterion = nn.CrossEntropyLoss()

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(0.05, 20.0)


def apply_temperature(logits: Tensor, temperature: Tensor | float) -> Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def entropy_percentile_threshold(entropies: Tensor, percentile: float) -> Tensor:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between zero and one hundred")
    return torch.quantile(entropies.float(), percentile / 100.0)


def out_of_distribution_score(
    embeddings: Tensor,
    reference_mean: Tensor,
    reference_inverse_covariance: Tensor,
) -> Tensor:
    difference = embeddings - reference_mean
    projected = torch.matmul(difference, reference_inverse_covariance)
    squared = torch.sum(projected * difference, dim=-1)
    return torch.sqrt(squared.clamp_min(0.0))


def covariance_precision(embeddings: Tensor, shrinkage: float = 0.01) -> tuple[Tensor, Tensor]:
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    covariance = torch.matmul(centered.transpose(0, 1), centered) / max(embeddings.shape[0] - 1, 1)
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    trace_scale = covariance.trace() / covariance.shape[0]
    regularized = (1.0 - shrinkage) * covariance + shrinkage * trace_scale * identity
    return mean, torch.linalg.pinv(regularized)

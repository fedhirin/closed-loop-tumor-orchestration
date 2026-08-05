from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from tumor_orchestration.metrics import (
    CalibrationResult,
    ConfidenceInterval,
    CurvePoint,
    HeterogeneityResult,
    ReclassificationResult,
    benjamini_hochberg,
    binary_auc,
    bonferroni,
    bootstrap_interval,
    calibration_metrics,
    concordance_index,
    decision_curve,
    dice_coefficient,
    macro_auc,
    net_reclassification_improvement,
    random_effects_meta_analysis,
    treatment_concordance,
)


class TaskPredictions(NamedTuple):
    response_target: np.ndarray
    response_probability: np.ndarray
    survival_time: np.ndarray
    survival_event: np.ndarray
    survival_risk: np.ndarray
    toxicity_target: np.ndarray
    toxicity_probability: np.ndarray
    segmentation_target: np.ndarray
    segmentation_probability: np.ndarray
    treatment_target: np.ndarray
    treatment_action: np.ndarray


@dataclass(frozen=True)
class PrimaryResults:
    response_auc: float
    survival_c_index: float
    toxicity_auc: float
    dice: float
    treatment_concordance: float


@dataclass(frozen=True)
class SubgroupResults:
    name: str
    size: int
    response_auc: float
    response_interval: ConfidenceInterval
    survival_c_index: float
    toxicity_auc: float
    dice: float
    concordance: float
    calibration: CalibrationResult


@dataclass(frozen=True)
class DatabaseResults:
    database: str
    size: int
    results: PrimaryResults
    response_interval: ConfidenceInterval
    response_standard_error: float


@dataclass(frozen=True)
class ClinicalUtilityResults:
    curve: tuple[CurvePoint, ...]
    reclassification: ReclassificationResult
    calibration: CalibrationResult
    positive_range: tuple[float, float] | None
    maximum_benefit: CurvePoint


@dataclass(frozen=True)
class CrossDatabaseResults:
    databases: tuple[DatabaseResults, ...]
    heterogeneity: HeterogeneityResult


def compute_primary(predictions: TaskPredictions) -> PrimaryResults:
    return PrimaryResults(
        binary_auc(predictions.response_target, predictions.response_probability),
        concordance_index(
            predictions.survival_time,
            predictions.survival_risk,
            predictions.survival_event,
        ),
        macro_auc(predictions.toxicity_target, predictions.toxicity_probability),
        dice_coefficient(
            predictions.segmentation_target,
            predictions.segmentation_probability,
        ),
        treatment_concordance(
            predictions.treatment_target,
            predictions.treatment_action,
        ),
    )


def subset_predictions(predictions: TaskPredictions, selected: np.ndarray) -> TaskPredictions:
    return TaskPredictions(*(field[selected] for field in predictions))


def response_interval(
    targets: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> ConfidenceInterval:
    pairs = tuple(zip(targets.tolist(), probabilities.tolist(), strict=True))

    def statistic(sample: Sequence[tuple[float, float]]) -> float:
        target = np.asarray([item[0] for item in sample])
        probability = np.asarray([item[1] for item in sample])
        return binary_auc(target, probability)

    return bootstrap_interval(pairs, statistic, iterations, seed=seed)


def analyze_subgroups(
    predictions: TaskPredictions,
    groups: Mapping[str, np.ndarray],
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[SubgroupResults, ...]:
    results: list[SubgroupResults] = []
    for offset, (name, selected) in enumerate(groups.items()):
        subset = subset_predictions(predictions, selected)
        primary = compute_primary(subset)
        interval = response_interval(
            subset.response_target,
            subset.response_probability,
            iterations,
            seed + offset,
        )
        calibration = calibration_metrics(
            subset.response_target,
            subset.response_probability,
        )
        results.append(
            SubgroupResults(
                name,
                int(selected.sum()),
                primary.response_auc,
                interval,
                primary.survival_c_index,
                primary.toxicity_auc,
                primary.dice,
                primary.treatment_concordance,
                calibration,
            )
        )
    return tuple(results)


def auc_standard_error(auc: float, positive: int, negative: int) -> float:
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    numerator = (
        auc * (1.0 - auc) + (positive - 1) * (q1 - auc * auc) + (negative - 1) * (q2 - auc * auc)
    )
    return float(np.sqrt(numerator / max(positive * negative, 1)))


def analyze_databases(
    predictions: TaskPredictions,
    databases: Mapping[str, np.ndarray],
    iterations: int = 1000,
    seed: int = 0,
) -> CrossDatabaseResults:
    results: list[DatabaseResults] = []
    for offset, (database, selected) in enumerate(databases.items()):
        subset = subset_predictions(predictions, selected)
        primary = compute_primary(subset)
        interval = response_interval(
            subset.response_target,
            subset.response_probability,
            iterations,
            seed + offset,
        )
        positive = int((subset.response_target == 1).sum())
        negative = int((subset.response_target == 0).sum())
        standard_error = auc_standard_error(primary.response_auc, positive, negative)
        results.append(
            DatabaseResults(database, int(selected.sum()), primary, interval, standard_error)
        )
    heterogeneity = random_effects_meta_analysis(
        [result.results.response_auc for result in results],
        [result.response_standard_error for result in results],
    )
    return CrossDatabaseResults(tuple(results), heterogeneity)


def clinical_utility(
    targets: np.ndarray,
    probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> ClinicalUtilityResults:
    selected_thresholds = (
        thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99).tolist()
    )
    curve = decision_curve(targets, probabilities, selected_thresholds)
    positive = [
        point.threshold
        for point in curve
        if point.model_benefit > point.treat_all_benefit
        and point.model_benefit > point.treat_none_benefit
    ]
    positive_range = (min(positive), max(positive)) if positive else None
    maximum = max(curve, key=lambda point: point.model_benefit)
    return ClinicalUtilityResults(
        curve,
        net_reclassification_improvement(targets, baseline_probabilities, probabilities),
        calibration_metrics(targets, probabilities),
        positive_range,
        maximum,
    )


def time_dependent_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
) -> float:
    cases = np.logical_and(event == 1, time <= horizon)
    controls = time > horizon
    selected = np.logical_or(cases, controls)
    if cases.sum() == 0 or controls.sum() == 0:
        return float("nan")
    return binary_auc(cases[selected].astype(int), risk[selected])


def survival_horizons(
    predictions: TaskPredictions,
    horizons: Sequence[float],
) -> dict[float, float]:
    return {
        float(horizon): time_dependent_auc(
            predictions.survival_time,
            predictions.survival_event,
            predictions.survival_risk,
            horizon,
        )
        for horizon in horizons
    }


def interaction_test(
    targets: np.ndarray,
    probabilities: np.ndarray,
    subgroup: np.ndarray,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    first = subgroup.astype(bool)
    observed = binary_auc(targets[first], probabilities[first]) - binary_auc(
        targets[~first],
        probabilities[~first],
    )
    generator = np.random.default_rng(seed)
    exceedances = 0
    valid = 0
    for _ in range(iterations):
        permuted = generator.permutation(subgroup).astype(bool)
        if np.unique(targets[permuted]).size < 2 or np.unique(targets[~permuted]).size < 2:
            continue
        difference = binary_auc(targets[permuted], probabilities[permuted]) - binary_auc(
            targets[~permuted],
            probabilities[~permuted],
        )
        exceedances += int(abs(difference) >= abs(observed))
        valid += 1
    return observed, (exceedances + 1.0) / (valid + 1.0)


def corrected_interactions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: Mapping[str, np.ndarray],
    correction: str = "benjamini-hochberg",
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    names = list(groups)
    raw = [
        interaction_test(
            targets,
            probabilities,
            groups[name],
            iterations,
            seed + index,
        )
        for index, name in enumerate(names)
    ]
    p_values = [item[1] for item in raw]
    adjusted = (
        benjamini_hochberg(p_values) if correction == "benjamini-hochberg" else bonferroni(p_values)
    )
    return {
        name: (raw[index][0], raw[index][1], float(adjusted[index]))
        for index, name in enumerate(names)
    }


def contraindication_rate(actions: np.ndarray, contraindicated: np.ndarray) -> float:
    rows = np.arange(actions.shape[0])
    return float(np.mean(contraindicated[rows, actions].astype(bool)))


def escalation_rate(escalated: np.ndarray) -> float:
    return float(np.mean(escalated.astype(bool)))


def switch_detection_accuracy(targets: np.ndarray, actions: np.ndarray) -> float:
    if targets.ndim != 2 or actions.ndim != 2:
        raise ValueError("switch sequences must have batch and cycle dimensions")
    target_switch = targets[:, 1:] != targets[:, :-1]
    action_switch = actions[:, 1:] != actions[:, :-1]
    return float(np.mean(target_switch == action_switch))


def switch_detection_latency(targets: np.ndarray, actions: np.ndarray) -> np.ndarray:
    latencies: list[int] = []
    for target_sequence, action_sequence in zip(targets, actions, strict=True):
        target_switches = np.flatnonzero(target_sequence[1:] != target_sequence[:-1]) + 1
        action_switches = np.flatnonzero(action_sequence[1:] != action_sequence[:-1]) + 1
        for target_index in target_switches:
            later = action_switches[action_switches >= target_index]
            if later.size:
                latencies.append(int(later[0] - target_index))
    return np.asarray(latencies, dtype=int)


def gaussian_noise_sweep(
    volumes: Tensor,
    levels: Sequence[float],
    evaluator: Callable[[Tensor], Mapping[str, float]],
    seed: int = 0,
) -> dict[float, Mapping[str, float]]:
    generator = torch.Generator(device=volumes.device)
    generator.manual_seed(seed)
    results: dict[float, Mapping[str, float]] = {}
    for level in levels:
        noise = torch.randn(volumes.shape, generator=generator, device=volumes.device)
        corrupted = volumes + noise * level
        results[float(level)] = evaluator(corrupted)
    return results


def label_noise_indices(size: int, rate: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    count = round(size * rate)
    return np.sort(generator.choice(size, size=count, replace=False))


def fgsm(inputs: Tensor, loss: Tensor, epsilon: float) -> Tensor:
    gradient = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
    return (inputs + epsilon * gradient.sign()).detach()


def projected_gradient_descent(
    inputs: Tensor,
    objective: Callable[[Tensor], Tensor],
    epsilon: float,
    steps: int = 10,
    step_size: float | None = None,
) -> Tensor:
    alpha = step_size if step_size is not None else epsilon / max(steps // 2, 1)
    adversarial = inputs.detach().clone()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = objective(adversarial)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        perturbation = (adversarial - inputs).clamp(-epsilon, epsilon)
        adversarial = (inputs + perturbation).detach()
    return adversarial


def save_results(path: Path, results: object) -> None:
    def convert(value: object) -> object:
        if hasattr(value, "__dataclass_fields__"):
            return {name: convert(item) for name, item in asdict(value).items()}
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(convert(results), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch import Tensor

T = TypeVar("T")


class ConfidenceInterval(NamedTuple):
    estimate: float
    lower: float
    upper: float


class CalibrationResult(NamedTuple):
    brier: float
    slope: float
    intercept: float
    expected_error: float
    maximum_error: float


class ReclassificationResult(NamedTuple):
    total: float
    event: float
    nonevent: float
    discrimination: float


class HeterogeneityResult(NamedTuple):
    cochran_q: float
    degrees_freedom: int
    p_value: float
    i_squared: float
    tau_squared: float
    pooled_effect: float
    pooled_standard_error: float


class CurvePoint(NamedTuple):
    threshold: float
    model_benefit: float
    treat_all_benefit: float
    treat_none_benefit: float


def _numpy(values: Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(values, Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def binary_auc(targets: Tensor | np.ndarray, scores: Tensor | np.ndarray) -> float:
    target_array = _numpy(targets).reshape(-1)
    score_array = _numpy(scores).reshape(-1)
    if np.unique(target_array).size < 2:
        return float("nan")
    return float(roc_auc_score(target_array, score_array))


def macro_auc(targets: Tensor | np.ndarray, scores: Tensor | np.ndarray) -> float:
    target_array = _numpy(targets)
    score_array = _numpy(scores)
    values: list[float] = []
    for column in range(target_array.shape[1]):
        if np.unique(target_array[:, column]).size >= 2:
            values.append(float(roc_auc_score(target_array[:, column], score_array[:, column])))
    return float(np.mean(values)) if values else float("nan")


def concordance_index(
    time: Tensor | np.ndarray,
    risk: Tensor | np.ndarray,
    event: Tensor | np.ndarray,
) -> float:
    times = _numpy(time).reshape(-1)
    risks = _numpy(risk).reshape(-1)
    events = _numpy(event).reshape(-1).astype(bool)
    concordant = 0.0
    comparable = 0.0
    for left in range(times.size):
        for right in range(left + 1, times.size):
            if times[left] == times[right]:
                continue
            if times[left] < times[right] and events[left]:
                comparable += 1.0
                if risks[left] > risks[right]:
                    concordant += 1.0
                elif risks[left] == risks[right]:
                    concordant += 0.5
            elif times[right] < times[left] and events[right]:
                comparable += 1.0
                if risks[right] > risks[left]:
                    concordant += 1.0
                elif risks[right] == risks[left]:
                    concordant += 0.5
    return concordant / comparable if comparable > 0 else float("nan")


def dice_coefficient(
    targets: Tensor | np.ndarray,
    predictions: Tensor | np.ndarray,
    threshold: float = 0.5,
    smoothing: float = 1.0,
) -> float:
    target_array = _numpy(targets).astype(bool).reshape(_numpy(targets).shape[0], -1)
    prediction_array = (_numpy(predictions) >= threshold).reshape(target_array.shape)
    intersection = np.logical_and(target_array, prediction_array).sum(axis=1)
    denominator = target_array.sum(axis=1) + prediction_array.sum(axis=1)
    return float(np.mean((2.0 * intersection + smoothing) / (denominator + smoothing)))


def treatment_concordance(targets: Tensor | np.ndarray, actions: Tensor | np.ndarray) -> float:
    return float(np.mean(_numpy(targets).reshape(-1) == _numpy(actions).reshape(-1)))


def brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities.astype(float) - targets.astype(float)) ** 2))


def calibration_slope(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    epsilon = np.finfo(float).eps
    bounded = np.clip(probabilities.astype(float), epsilon, 1.0 - epsilon)
    logits = np.log(bounded / (1.0 - bounded))
    design = np.column_stack((np.ones(logits.size), logits))
    coefficients = np.zeros(2, dtype=float)
    for _ in range(100):
        linear = design @ coefficients
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        gradient = design.T @ (targets - fitted)
        weights = fitted * (1.0 - fitted)
        hessian = -(design.T * weights) @ design
        update = np.linalg.pinv(hessian) @ gradient
        coefficients -= update
        if np.linalg.norm(update) < 1e-10:
            break
    return float(coefficients[1]), float(coefficients[0])


def calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[float, float]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    expected = 0.0
    maximum = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            selected = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        if not selected.any():
            continue
        difference = abs(float(targets[selected].mean() - probabilities[selected].mean()))
        expected += difference * float(selected.mean())
        maximum = max(maximum, difference)
    return expected, maximum


def calibration_metrics(
    targets: Tensor | np.ndarray,
    probabilities: Tensor | np.ndarray,
    bins: int = 10,
) -> CalibrationResult:
    target_array = _numpy(targets).reshape(-1).astype(float)
    probability_array = _numpy(probabilities).reshape(-1).astype(float)
    slope, intercept = calibration_slope(target_array, probability_array)
    expected, maximum = calibration_error(target_array, probability_array, bins)
    return CalibrationResult(
        brier_score(target_array, probability_array),
        slope,
        intercept,
        expected,
        maximum,
    )


def net_benefit(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    selected = probabilities >= threshold
    true_positive = np.logical_and(selected, targets == 1).sum()
    false_positive = np.logical_and(selected, targets == 0).sum()
    odds = threshold / max(1.0 - threshold, np.finfo(float).eps)
    return float(true_positive / targets.size - false_positive / targets.size * odds)


def treat_all_benefit(targets: np.ndarray, threshold: float) -> float:
    prevalence = float(np.mean(targets == 1))
    odds = threshold / max(1.0 - threshold, np.finfo(float).eps)
    return prevalence - (1.0 - prevalence) * odds


def decision_curve(
    targets: Tensor | np.ndarray,
    probabilities: Tensor | np.ndarray,
    thresholds: Iterable[float],
) -> tuple[CurvePoint, ...]:
    target_array = _numpy(targets).reshape(-1)
    probability_array = _numpy(probabilities).reshape(-1)
    points: list[CurvePoint] = []
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError("decision thresholds must be between zero and one")
        points.append(
            CurvePoint(
                float(threshold),
                net_benefit(target_array, probability_array, threshold),
                treat_all_benefit(target_array, threshold),
                0.0,
            )
        )
    return tuple(points)


def net_reclassification_improvement(
    targets: Tensor | np.ndarray,
    baseline: Tensor | np.ndarray,
    candidate: Tensor | np.ndarray,
) -> ReclassificationResult:
    target_array = _numpy(targets).reshape(-1).astype(bool)
    baseline_array = _numpy(baseline).reshape(-1)
    candidate_array = _numpy(candidate).reshape(-1)
    event_up = np.mean(candidate_array[target_array] > baseline_array[target_array])
    event_down = np.mean(candidate_array[target_array] < baseline_array[target_array])
    nonevent_down = np.mean(candidate_array[~target_array] < baseline_array[~target_array])
    nonevent_up = np.mean(candidate_array[~target_array] > baseline_array[~target_array])
    event = float(event_up - event_down)
    nonevent = float(nonevent_down - nonevent_up)
    discrimination = float(
        (candidate_array[target_array].mean() - candidate_array[~target_array].mean())
        - (baseline_array[target_array].mean() - baseline_array[~target_array].mean())
    )
    return ReclassificationResult(event + nonevent, event, nonevent, discrimination)


def bootstrap_interval(
    values: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if len(values) == 0:
        raise ValueError("bootstrap values cannot be empty")
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    array = np.asarray(values, dtype=object)
    for iteration in range(iterations):
        indices = generator.integers(0, len(values), len(values))
        estimates[iteration] = statistic(array[indices].tolist())
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        statistic(values),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def paired_bootstrap_interval(
    targets: np.ndarray,
    predictions: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if targets.shape[0] != predictions.shape[0]:
        raise ValueError("targets and predictions must have equal first dimension")
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = generator.integers(0, targets.shape[0], targets.shape[0])
        value = statistic(targets[indices], predictions[indices])
        if np.isfinite(value):
            estimates.append(value)
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        statistic(targets, predictions),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def delong_covariance(targets: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    positive = predictions[targets == 1]
    negative = predictions[targets == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan"), float("nan")
    comparisons = np.zeros((positive.size, negative.size), dtype=float)
    comparisons[positive[:, None] > negative[None, :]] = 1.0
    comparisons[positive[:, None] == negative[None, :]] = 0.5
    auc = float(comparisons.mean())
    positive_components = comparisons.mean(axis=1)
    negative_components = comparisons.mean(axis=0)
    variance = np.var(positive_components, ddof=1) / positive.size
    variance += np.var(negative_components, ddof=1) / negative.size
    return auc, float(variance)


def delong_test(
    targets: Tensor | np.ndarray,
    first: Tensor | np.ndarray,
    second: Tensor | np.ndarray,
) -> tuple[float, float]:
    target_array = _numpy(targets).reshape(-1)
    first_array = _numpy(first).reshape(-1)
    second_array = _numpy(second).reshape(-1)
    positive = target_array == 1
    negative = target_array == 0
    first_matrix = np.where(
        first_array[positive, None] > first_array[negative][None, :],
        1.0,
        np.where(first_array[positive, None] == first_array[negative][None, :], 0.5, 0.0),
    )
    second_matrix = np.where(
        second_array[positive, None] > second_array[negative][None, :],
        1.0,
        np.where(second_array[positive, None] == second_array[negative][None, :], 0.5, 0.0),
    )
    differences_positive = first_matrix.mean(axis=1) - second_matrix.mean(axis=1)
    differences_negative = first_matrix.mean(axis=0) - second_matrix.mean(axis=0)
    variance = np.var(differences_positive, ddof=1) / max(differences_positive.size, 1)
    variance += np.var(differences_negative, ddof=1) / max(differences_negative.size, 1)
    difference = float(first_matrix.mean() - second_matrix.mean())
    z_score = difference / math.sqrt(max(float(variance), np.finfo(float).eps))
    p_value = float(2.0 * stats.norm.sf(abs(z_score)))
    return z_score, p_value


def mcnemar_test(first_correct: np.ndarray, second_correct: np.ndarray) -> tuple[float, float]:
    first_only = int(np.logical_and(first_correct, ~second_correct).sum())
    second_only = int(np.logical_and(~first_correct, second_correct).sum())
    discordant = first_only + second_only
    if discordant == 0:
        return 0.0, 1.0
    statistic = (abs(first_only - second_only) - 1.0) ** 2 / discordant
    return float(statistic), float(stats.chi2.sf(statistic, 1))


def paired_t_test(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    statistic, p_value = stats.ttest_rel(np.asarray(first), np.asarray(second))
    return float(statistic), float(p_value)


def bonferroni(p_values: Sequence[float], comparisons: int | None = None) -> np.ndarray:
    count = comparisons if comparisons is not None else len(p_values)
    return np.minimum(np.asarray(p_values, dtype=float) * count, 1.0)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def random_effects_meta_analysis(
    effects: Sequence[float],
    standard_errors: Sequence[float],
) -> HeterogeneityResult:
    effect_array = np.asarray(effects, dtype=float)
    error_array = np.asarray(standard_errors, dtype=float)
    if effect_array.size != error_array.size:
        raise ValueError("effects and standard errors must have equal length")
    fixed_weights = 1.0 / np.square(error_array)
    fixed_effect = np.sum(fixed_weights * effect_array) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * np.square(effect_array - fixed_effect)))
    degrees = effect_array.size - 1
    correction = np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    tau_squared = max(0.0, (q - degrees) / correction)
    random_weights = 1.0 / (np.square(error_array) + tau_squared)
    pooled = float(np.sum(random_weights * effect_array) / np.sum(random_weights))
    pooled_error = float(math.sqrt(1.0 / np.sum(random_weights)))
    i_squared = max(0.0, (q - degrees) / q * 100.0) if q > 0.0 else 0.0
    p_value = float(stats.chi2.sf(q, degrees))
    return HeterogeneityResult(
        q,
        degrees,
        p_value,
        i_squared,
        tau_squared,
        pooled,
        pooled_error,
    )


def permutation_synergy_test(
    full_gain: float,
    component_gains: Sequence[float],
    samples: np.ndarray,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    additive = float(np.sum(component_gains))
    synergy = full_gain / additive if additive != 0.0 else float("nan")
    generator = np.random.default_rng(seed)
    exceedances = 0
    observed = full_gain - additive
    for _ in range(iterations):
        signs = generator.choice((-1.0, 1.0), size=samples.shape)
        permuted = float(np.mean(samples * signs))
        if permuted >= observed:
            exceedances += 1
    return synergy, (exceedances + 1.0) / (iterations + 1.0)


@dataclass
class MetricAccumulator:
    response_targets: list[Tensor]
    response_scores: list[Tensor]
    survival_times: list[Tensor]
    survival_events: list[Tensor]
    survival_risks: list[Tensor]
    toxicity_targets: list[Tensor]
    toxicity_scores: list[Tensor]
    segmentation_targets: list[Tensor]
    segmentation_scores: list[Tensor]
    treatment_targets: list[Tensor]
    treatment_actions: list[Tensor]

    @classmethod
    def create(cls) -> MetricAccumulator:
        return cls([], [], [], [], [], [], [], [], [], [], [])

    def update(
        self,
        response_targets: Tensor,
        response_scores: Tensor,
        survival_times: Tensor,
        survival_events: Tensor,
        survival_risks: Tensor,
        toxicity_targets: Tensor,
        toxicity_scores: Tensor,
        segmentation_targets: Tensor,
        segmentation_scores: Tensor,
        treatment_targets: Tensor,
        treatment_actions: Tensor,
    ) -> None:
        self.response_targets.append(response_targets.detach().cpu())
        self.response_scores.append(response_scores.detach().cpu())
        self.survival_times.append(survival_times.detach().cpu())
        self.survival_events.append(survival_events.detach().cpu())
        self.survival_risks.append(survival_risks.detach().cpu())
        self.toxicity_targets.append(toxicity_targets.detach().cpu())
        self.toxicity_scores.append(toxicity_scores.detach().cpu())
        self.segmentation_targets.append(segmentation_targets.detach().cpu())
        self.segmentation_scores.append(segmentation_scores.detach().cpu())
        self.treatment_targets.append(treatment_targets.detach().cpu())
        self.treatment_actions.append(treatment_actions.detach().cpu())

    def compute(self) -> dict[str, float]:
        response_targets = torch.cat(self.response_targets)
        response_scores = torch.cat(self.response_scores)
        survival_times = torch.cat(self.survival_times)
        survival_events = torch.cat(self.survival_events)
        survival_risks = torch.cat(self.survival_risks)
        toxicity_targets = torch.cat(self.toxicity_targets)
        toxicity_scores = torch.cat(self.toxicity_scores)
        segmentation_targets = torch.cat(self.segmentation_targets)
        segmentation_scores = torch.cat(self.segmentation_scores)
        treatment_targets = torch.cat(self.treatment_targets)
        treatment_actions = torch.cat(self.treatment_actions)
        return {
            "response_auc": binary_auc(response_targets, response_scores),
            "survival_c_index": concordance_index(
                survival_times,
                survival_risks,
                survival_events,
            ),
            "toxicity_auc": macro_auc(toxicity_targets, toxicity_scores),
            "dice": dice_coefficient(segmentation_targets, segmentation_scores),
            "treatment_concordance": treatment_concordance(
                treatment_targets,
                treatment_actions,
            ),
        }

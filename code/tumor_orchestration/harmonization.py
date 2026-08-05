from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class StandardizationState:
    mean: np.ndarray
    scale: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class CombatState:
    grand_mean: np.ndarray
    pooled_variance: np.ndarray
    batch_location: np.ndarray
    batch_scale: np.ndarray
    batches: tuple[str, ...]


def fit_standardization(values: np.ndarray) -> StandardizationState:
    observed = np.isfinite(values)
    counts = observed.sum(axis=0)
    totals = np.where(observed, values, 0.0).sum(axis=0)
    mean = totals / np.maximum(counts, 1)
    centered = np.where(observed, values - mean, 0.0)
    variance = np.square(centered).sum(axis=0) / np.maximum(counts - 1, 1)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return StandardizationState(mean.astype(np.float32), scale.astype(np.float32), observed)


def apply_standardization(values: np.ndarray, state: StandardizationState) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, state.mean)
    return ((filled - state.mean) / state.scale).astype(np.float32)


def quantile_reference(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, axis=0)
    return np.nanmean(sorted_values, axis=1).astype(np.float32)


def apply_quantile_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if values.shape[0] != reference.shape[0]:
        raise ValueError("reference length must equal the feature dimension")
    order = np.argsort(values, axis=0)
    normalized = np.empty_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        normalized[order[:, column], column] = reference
    return normalized


def design_matrix(covariates: np.ndarray | None, batches: Sequence[str]) -> np.ndarray:
    unique = tuple(dict.fromkeys(batches))
    batch_matrix = np.zeros((len(batches), len(unique)), dtype=np.float64)
    lookup = {name: index for index, name in enumerate(unique)}
    for row, batch in enumerate(batches):
        batch_matrix[row, lookup[batch]] = 1.0
    if covariates is None:
        return batch_matrix
    if covariates.shape[0] != len(batches):
        raise ValueError("covariates and batches must have equal sample counts")
    return np.column_stack((batch_matrix, covariates.astype(np.float64)))


def fit_combat(
    values: np.ndarray,
    batches: Sequence[str],
    covariates: np.ndarray | None = None,
) -> CombatState:
    if values.shape[0] != len(batches):
        raise ValueError("values and batches must have equal sample counts")
    unique = tuple(dict.fromkeys(batches))
    design = design_matrix(covariates, batches)
    coefficients = np.linalg.pinv(design) @ values
    batch_coefficients = coefficients[: len(unique)]
    counts = np.asarray([batches.count(name) for name in unique], dtype=np.float64)
    grand_mean = counts @ batch_coefficients / counts.sum()
    fitted_covariates = 0.0
    if covariates is not None:
        fitted_covariates = covariates @ coefficients[len(unique) :]
    residual = values - np.asarray(fitted_covariates) - grand_mean
    pooled_variance = np.mean(np.square(residual), axis=0)
    standardized = residual / np.sqrt(np.maximum(pooled_variance, 1e-8))
    location = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    scale = np.ones_like(location)
    for index, name in enumerate(unique):
        selected = np.asarray([item == name for item in batches])
        location[index] = standardized[selected].mean(axis=0)
        scale[index] = standardized[selected].var(axis=0, ddof=1)
    location_prior = location.mean(axis=1)
    location_variance = location.var(axis=1, ddof=1)
    scale_mean = scale.mean(axis=1)
    scale_variance = scale.var(axis=1, ddof=1)
    adjusted_location = np.zeros_like(location)
    adjusted_scale = np.ones_like(scale)
    for index, name in enumerate(unique):
        selected = np.asarray([item == name for item in batches])
        sample_count = max(int(selected.sum()), 1)
        numerator = location_variance[index] * sample_count * location[index]
        numerator += scale[index] * location_prior[index]
        denominator = location_variance[index] * sample_count + scale[index]
        adjusted_location[index] = numerator / np.maximum(denominator, 1e-8)
        shape = (2.0 * scale_variance[index] + scale_mean[index] ** 2) / max(
            scale_variance[index],
            1e-8,
        )
        rate = (scale_mean[index] * scale_variance[index] + scale_mean[index] ** 3) / max(
            scale_variance[index],
            1e-8,
        )
        sum_square = np.square(standardized[selected] - adjusted_location[index]).sum(axis=0)
        adjusted_scale[index] = (rate + 0.5 * sum_square) / (shape + sample_count / 2.0 - 1.0)
    return CombatState(
        grand_mean.astype(np.float32),
        pooled_variance.astype(np.float32),
        adjusted_location.astype(np.float32),
        adjusted_scale.astype(np.float32),
        unique,
    )


def apply_combat(values: np.ndarray, batches: Sequence[str], state: CombatState) -> np.ndarray:
    if values.shape[0] != len(batches):
        raise ValueError("values and batches must have equal sample counts")
    standardized = (values - state.grand_mean) / np.sqrt(np.maximum(state.pooled_variance, 1e-8))
    lookup = {name: index for index, name in enumerate(state.batches)}
    adjusted = np.empty_like(standardized)
    for row, name in enumerate(batches):
        if name not in lookup:
            adjusted[row] = standardized[row]
            continue
        index = lookup[name]
        adjusted[row] = (standardized[row] - state.batch_location[index]) / np.sqrt(
            np.maximum(state.batch_scale[index], 1e-8)
        )
    restored = adjusted * np.sqrt(np.maximum(state.pooled_variance, 1e-8)) + state.grand_mean
    return restored.astype(np.float32)


def masked_mean(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    total = torch.sum(values * weights, dim=dimension)
    count = torch.sum(weights, dim=dimension).clamp_min(1.0)
    return total / count


def masked_variance(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    mean = masked_mean(values, mask, dimension)
    expanded = mean.unsqueeze(dimension)
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    squared = torch.square(values - expanded) * weights
    count = torch.sum(weights, dim=dimension).clamp_min(2.0)
    return torch.sum(squared, dim=dimension) / (count - 1.0)


def masked_standardize(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    mean = masked_mean(values, mask, dimension).unsqueeze(dimension)
    variance = masked_variance(values, mask, dimension).unsqueeze(dimension)
    normalized = (values - mean) / torch.sqrt(variance + 1e-6)
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return normalized * weights


def rank_gaussian(values: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata

    result = np.empty_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        ranks = rankdata(values[:, column], method="average")
        quantiles = (ranks - 0.5) / len(ranks)
        result[:, column] = norm.ppf(quantiles).astype(np.float32)
    return result


def winsorize(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    low = np.quantile(values, lower, axis=0)
    high = np.quantile(values, upper, axis=0)
    return np.clip(values, low, high).astype(np.float32)


def median_absolute_deviation(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    deviation = np.median(np.abs(values - median), axis=0)
    return deviation.astype(np.float32)


def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    deviation = median_absolute_deviation(values)
    scale = np.where(deviation > 1e-8, deviation * 1.4826, 1.0)
    return ((values - median) / scale).astype(np.float32)

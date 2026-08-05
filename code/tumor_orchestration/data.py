from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from tumor_orchestration.records import CohortSplit, PatientTrajectory

T = TypeVar("T")


@dataclass(frozen=True)
class CTSeriesRecord:
    patient_id: str
    cycle: int
    volume_path: Path
    mask_path: Path | None
    spacing: tuple[float, float, float]
    elapsed_hours: float


@dataclass(frozen=True)
class GenomicRecord:
    patient_id: str
    expression: np.ndarray
    mutations: np.ndarray


@dataclass(frozen=True)
class EHREvent:
    patient_id: str
    cycle: int
    timestamp_hours: float
    values: np.ndarray


@dataclass(frozen=True)
class DrugGraphRecord:
    patient_id: str
    cycle: int
    node_features: np.ndarray
    adjacency: np.ndarray
    treatment_index: int


@dataclass(frozen=True)
class OutcomeRecord:
    patient_id: str
    response: float
    survival_time: float
    survival_event: float
    toxicity: np.ndarray
    action: int


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    size: int
    sha256: str


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, paths: Iterable[Path]) -> tuple[ManifestEntry, ...]:
    entries: list[ManifestEntry] = []
    for path in sorted(paths):
        entries.append(
            ManifestEntry(
                str(path.relative_to(root)),
                path.stat().st_size,
                sha256_file(path),
            )
        )
    return tuple(entries)


def save_manifest(path: Path, entries: Sequence[ManifestEntry]) -> None:
    payload = [entry.__dict__ for entry in entries]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def verify_manifest(root: Path, entries: Sequence[ManifestEntry]) -> tuple[str, ...]:
    failures: list[str] = []
    for entry in entries:
        path = root / entry.relative_path
        if not path.is_file():
            failures.append(f"missing:{entry.relative_path}")
            continue
        if path.stat().st_size != entry.size:
            failures.append(f"size:{entry.relative_path}")
            continue
        if sha256_file(path) != entry.sha256:
            failures.append(f"hash:{entry.relative_path}")
    return tuple(failures)


def lung_window(volume: np.ndarray, low: float = -1024.0, high: float = 400.0) -> np.ndarray:
    clipped = np.clip(volume.astype(np.float32), low, high)
    return (clipped - low) / (high - low)


def dual_window(volume: np.ndarray) -> np.ndarray:
    lung = np.clip(volume, -1350.0, 150.0)
    mediastinal = np.clip(volume, -160.0, 240.0)
    lung = (lung + 1350.0) / 1500.0
    mediastinal = (mediastinal + 160.0) / 400.0
    return np.stack((lung, mediastinal), axis=0).astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    source_spacing: Sequence[float],
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> np.ndarray:
    if len(source_spacing) != 3 or len(target_spacing) != 3:
        raise ValueError("spacing must contain three values")
    zoom = tuple(
        float(source) / float(target)
        for source, target in zip(source_spacing, target_spacing, strict=True)
    )
    return np.asarray(ndimage.zoom(volume, zoom=zoom, order=order), dtype=np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask > 0)
    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask > 0, labels, range(1, count + 1))
    selected = int(np.argmax(sizes)) + 1
    return (labels == selected).astype(np.uint8)


def lesion_center(mask: np.ndarray) -> tuple[int, int, int]:
    component = largest_component(mask)
    coordinates = np.argwhere(component > 0)
    if coordinates.size == 0:
        return tuple(int(size // 2) for size in mask.shape)
    center = np.round(coordinates.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def centered_crop(
    volume: np.ndarray,
    center: Sequence[int],
    shape: Sequence[int],
    fill: float = 0.0,
) -> np.ndarray:
    if len(center) != volume.ndim or len(shape) != volume.ndim:
        raise ValueError("center and shape must match volume rank")
    output = np.full(tuple(int(item) for item in shape), fill, dtype=volume.dtype)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for axis, requested in enumerate(shape):
        start = int(center[axis]) - int(requested) // 2
        end = start + int(requested)
        source_start = max(start, 0)
        source_end = min(end, volume.shape[axis])
        target_start = source_start - start
        target_end = target_start + source_end - source_start
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_end))
    output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def random_affine_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    generator: np.random.Generator,
    maximum_rotation: float = 8.0,
    maximum_shift: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    angle = float(generator.uniform(-maximum_rotation, maximum_rotation))
    axes_options = ((0, 1), (0, 2), (1, 2))
    axes = axes_options[int(generator.integers(0, len(axes_options)))]
    rotated_volume = ndimage.rotate(
        volume, angle, axes=axes, reshape=False, order=1, mode="nearest"
    )
    rotated_mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0, mode="constant")
    shifts = generator.uniform(-maximum_shift, maximum_shift, size=3)
    shifted_volume = ndimage.shift(rotated_volume, shifts, order=1, mode="nearest")
    shifted_mask = ndimage.shift(rotated_mask, shifts, order=0, mode="constant")
    return shifted_volume.astype(np.float32), (shifted_mask > 0.5).astype(np.float32)


def standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale > 1e-8, scale, 1.0)
    return ((values - mean) / safe_scale).astype(np.float32)


def log_quantile_normalize(matrix: np.ndarray) -> np.ndarray:
    logged = np.log2(np.maximum(matrix, 0.0) + 1.0)
    order = np.argsort(logged, axis=0)
    sorted_values = np.sort(logged, axis=0)
    reference = sorted_values.mean(axis=1)
    normalized = np.empty_like(logged)
    for column in range(logged.shape[1]):
        normalized[order[:, column], column] = reference
    return normalized.astype(np.float32)


def prevalence_filter(
    mutations: np.ndarray, minimum: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    prevalence = np.mean(mutations > 0, axis=0)
    keep = prevalence >= minimum
    return mutations[:, keep].astype(np.float32), keep


def genomic_dropout(values: Tensor, probability: float, training: bool = True) -> Tensor:
    if not training or probability <= 0.0:
        return values
    keep = torch.rand_like(values) >= probability
    return values * keep / max(1.0 - probability, 1e-8)


def temporal_jitter(times: Tensor, hours: float, training: bool = True) -> Tensor:
    if not training or hours <= 0.0:
        return times
    return times + torch.empty_like(times).uniform_(-hours, hours)


class ChainedEquationImputer:
    def __init__(self, imputations: int = 5, iterations: int = 10, seed: int = 0) -> None:
        self.imputations = imputations
        self.iterations = iterations
        self.seed = seed
        self.means: np.ndarray | None = None
        self.coefficients: list[tuple[np.ndarray, float]] = []

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        matrix = values.astype(np.float64, copy=True)
        missing = np.isnan(matrix)
        means = np.nanmean(matrix, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        matrix[missing] = np.take(means, np.where(missing)[1])
        generator = np.random.default_rng(self.seed)
        completed: list[np.ndarray] = []
        for _ in range(self.imputations):
            current = matrix.copy()
            for _ in range(self.iterations):
                for feature in range(current.shape[1]):
                    missing_feature = missing[:, feature]
                    observed_feature = ~missing_feature
                    if not missing_feature.any() or observed_feature.sum() < 2:
                        continue
                    predictors = np.delete(current, feature, axis=1)
                    observed_x = predictors[observed_feature]
                    observed_y = current[observed_feature, feature]
                    design = np.column_stack((observed_x, np.ones(observed_x.shape[0])))
                    coefficients, _, _, _ = np.linalg.lstsq(design, observed_y, rcond=None)
                    predicted = (
                        np.column_stack(
                            (predictors[missing_feature], np.ones(missing_feature.sum()))
                        )
                        @ coefficients
                    )
                    residual = observed_y - design @ coefficients
                    noise = generator.normal(0.0, residual.std() + 1e-8, predicted.shape)
                    current[missing_feature, feature] = predicted + noise
            completed.append(current)
        self.means = means
        return np.mean(completed, axis=0).astype(np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("imputer has not been fitted")
        result = values.astype(np.float32, copy=True)
        missing = np.isnan(result)
        result[missing] = np.take(self.means, np.where(missing)[1])
        return result


def patient_level_split(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    validation_fraction: float,
    testing_fraction: float,
    seed: int,
) -> CohortSplit:
    if len(patient_ids) != len(labels):
        raise ValueError("patient identifiers and labels must have equal length")
    groups: dict[int, list[str]] = defaultdict(list)
    for patient_id, label in zip(patient_ids, labels, strict=True):
        groups[int(label)].append(patient_id)
    generator = random.Random(seed)
    training: list[str] = []
    validation: list[str] = []
    testing: list[str] = []
    for group in groups.values():
        generator.shuffle(group)
        test_count = round(len(group) * testing_fraction)
        validation_count = round(len(group) * validation_fraction)
        testing.extend(group[:test_count])
        validation.extend(group[test_count : test_count + validation_count])
        training.extend(group[test_count + validation_count :])
    split = CohortSplit(tuple(training), tuple(validation), tuple(testing))
    split.validate()
    return split


def stratified_folds(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    folds: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    groups: dict[int, list[str]] = defaultdict(list)
    for patient_id, label in zip(patient_ids, labels, strict=True):
        groups[int(label)].append(patient_id)
    generator = random.Random(seed)
    partitions: list[list[str]] = [[] for _ in range(folds)]
    for group in groups.values():
        generator.shuffle(group)
        for index, patient_id in enumerate(group):
            partitions[index % folds].append(patient_id)
    return tuple(tuple(partition) for partition in partitions)


class PatientBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        patient_ids: Sequence[str],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, patient_id in enumerate(patient_ids):
            grouped[patient_id].append(index)
        self.groups = tuple(tuple(indices) for indices in grouped.values())
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        groups = list(self.groups)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        batch: list[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(group) for group in self.groups)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)


class TrajectoryDataset(Dataset[PatientTrajectory]):
    def __init__(self, trajectories: Sequence[PatientTrajectory]) -> None:
        self.trajectories = tuple(trajectories)
        identifiers = [trajectory.patient_id for trajectory in trajectories]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("trajectory identifiers must be unique")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> PatientTrajectory:
        return self.trajectories[index]


def pad_tensor_sequence(items: Sequence[Tensor], length: int, value: float = 0.0) -> Tensor:
    if len(items) == 0:
        raise ValueError("cannot pad an empty sequence")
    shape = (len(items), length, *items[0].shape[1:])
    output = items[0].new_full(shape, value)
    for index, item in enumerate(items):
        count = min(item.shape[0], length)
        output[index, :count] = item[:count]
    return output


def collate_trajectories(trajectories: Sequence[PatientTrajectory]) -> dict[str, Tensor]:
    if len(trajectories) == 0:
        raise ValueError("cannot collate an empty trajectory list")
    maximum_cycles = max(len(trajectory) for trajectory in trajectories)
    valid = torch.zeros(len(trajectories), maximum_cycles, dtype=torch.bool)
    ct: list[Tensor] = []
    expression: list[Tensor] = []
    mutations: list[Tensor] = []
    ehr: list[Tensor] = []
    ehr_times: list[Tensor] = []
    drug_nodes: list[Tensor] = []
    drug_adjacency: list[Tensor] = []
    treatment: list[Tensor] = []
    modality_mask: list[Tensor] = []
    for patient_index, trajectory in enumerate(trajectories):
        valid[patient_index, : len(trajectory)] = True
        cycles = trajectory.cycles
        ct.append(torch.stack([cycle.ct for cycle in cycles]))
        expression.append(torch.stack([cycle.expression for cycle in cycles]))
        mutations.append(torch.stack([cycle.mutations for cycle in cycles]))
        ehr.append(torch.stack([cycle.ehr for cycle in cycles]))
        ehr_times.append(torch.stack([cycle.ehr_times for cycle in cycles]))
        drug_nodes.append(torch.stack([cycle.drug_nodes for cycle in cycles]))
        drug_adjacency.append(torch.stack([cycle.drug_adjacency for cycle in cycles]))
        treatment.append(torch.stack([cycle.treatment for cycle in cycles]))
        modality_mask.append(torch.stack([cycle.modality_mask for cycle in cycles]))
    targets = [trajectory.targets for trajectory in trajectories]
    return {
        "ct": pad_tensor_sequence(ct, maximum_cycles),
        "expression": pad_tensor_sequence(expression, maximum_cycles),
        "mutations": pad_tensor_sequence(mutations, maximum_cycles),
        "ehr": pad_tensor_sequence(ehr, maximum_cycles),
        "ehr_times": pad_tensor_sequence(ehr_times, maximum_cycles),
        "drug_nodes": pad_tensor_sequence(drug_nodes, maximum_cycles),
        "drug_adjacency": pad_tensor_sequence(drug_adjacency, maximum_cycles),
        "treatment": pad_tensor_sequence(treatment, maximum_cycles),
        "modality_mask": pad_tensor_sequence(modality_mask, maximum_cycles),
        "valid_cycles": valid,
        "response": torch.stack([target.response for target in targets]),
        "survival_time": torch.stack([target.survival_time for target in targets]),
        "survival_event": torch.stack([target.survival_event for target in targets]),
        "toxicity": torch.stack([target.toxicity for target in targets]),
        "segmentation": torch.stack([target.segmentation for target in targets]),
        "actions": torch.stack([target.actions for target in targets]),
        "advantages": torch.stack([target.advantages for target in targets]),
        "old_log_probabilities": torch.stack([target.old_log_probabilities for target in targets]),
    }


def read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def group_by(items: Iterable[T], key: Callable[[T], str]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return dict(grouped)


def filter_lung_cancer_codes(rows: Iterable[Mapping[str, str]]) -> tuple[Mapping[str, str], ...]:
    selected: list[Mapping[str, str]] = []
    for row in rows:
        code = row.get("icd_code", "").replace(".", "").upper()
        version = row.get("icd_version", "10")
        if version == "10" and code.startswith("C34"):
            selected.append(row)
    return tuple(selected)

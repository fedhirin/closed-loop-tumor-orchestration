from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClinicalPatient:
    identifier: str
    age: float
    sex: str
    histology: str
    stage: str
    vital_status: str
    survival_days: float


@dataclass(frozen=True)
class MutationCall:
    patient_id: str
    gene: str
    chromosome: str
    position: int
    reference: str
    alternate: str
    classification: str


@dataclass(frozen=True)
class ExpressionProfile:
    patient_id: str
    genes: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class TimelineEvent:
    patient_id: str
    timestamp: datetime
    category: str
    code: str
    numeric_value: float | None
    unit: str | None


@dataclass(frozen=True)
class TreatmentCycle:
    patient_id: str
    index: int
    started: datetime
    ended: datetime | None
    regimen: tuple[str, ...]
    response: str | None


@dataclass(frozen=True)
class PathwayDefinition:
    identifier: str
    name: str
    genes: frozenset[str]


def open_text(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield from stream
    else:
        with path.open("r", encoding="utf-8") as stream:
            yield from stream


def parse_delimited(path: Path, delimiter: str = "\t") -> Iterator[dict[str, str]]:
    stream = open_text(path)
    reader = csv.DictReader(stream, delimiter=delimiter)
    for row in reader:
        yield dict(row)


def normalize_patient_identifier(identifier: str) -> str:
    return identifier.strip().upper().replace("_", "-")


def tcga_patient_identifier(barcode: str) -> str:
    fields = normalize_patient_identifier(barcode).split("-")
    return "-".join(fields[:3])


def parse_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value.strip().lower() in {"", "na", "n/a", "null", "not reported"}:
        return default
    return float(value)


def parse_date(value: str | None) -> datetime | None:
    if value is None or value.strip() == "":
        return None
    candidates = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y")
    for pattern in candidates:
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    raise ValueError(f"unsupported timestamp format: {value}")


def read_tcga_clinical(path: Path) -> tuple[ClinicalPatient, ...]:
    patients: list[ClinicalPatient] = []
    for row in parse_delimited(path):
        identifier = tcga_patient_identifier(
            row.get("case_submitter_id", row.get("bcr_patient_barcode", ""))
        )
        age_days = parse_float(row.get("age_at_diagnosis"))
        age = age_days / 365.25 if np.isfinite(age_days) and age_days > 200 else age_days
        death = parse_float(row.get("days_to_death"))
        follow_up = parse_float(row.get("days_to_last_follow_up"))
        survival = death if np.isfinite(death) else follow_up
        patients.append(
            ClinicalPatient(
                identifier,
                age,
                row.get("gender", "unknown").lower(),
                row.get("primary_diagnosis", row.get("histological_type", "unknown")),
                row.get("ajcc_pathologic_stage", row.get("tumor_stage", "unknown")),
                row.get("vital_status", "unknown").lower(),
                survival,
            )
        )
    return tuple(patients)


def read_maf(path: Path) -> tuple[MutationCall, ...]:
    calls: list[MutationCall] = []
    for row in parse_delimited(path):
        barcode = row.get("Tumor_Sample_Barcode", row.get("case_submitter_id", ""))
        calls.append(
            MutationCall(
                tcga_patient_identifier(barcode),
                row.get("Hugo_Symbol", ""),
                row.get("Chromosome", ""),
                int(parse_float(row.get("Start_Position"), 0.0)),
                row.get("Reference_Allele", ""),
                row.get("Tumor_Seq_Allele2", ""),
                row.get("Variant_Classification", "unknown"),
            )
        )
    return tuple(calls)


def mutation_matrix(
    patient_ids: Sequence[str],
    calls: Iterable[MutationCall],
    genes: Sequence[str],
) -> np.ndarray:
    patient_index = {identifier: index for index, identifier in enumerate(patient_ids)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    matrix = np.zeros((len(patient_ids), len(genes)), dtype=np.float32)
    for call in calls:
        row = patient_index.get(call.patient_id)
        column = gene_index.get(call.gene)
        if row is not None and column is not None:
            matrix[row, column] = 1.0
    return matrix


def read_expression_matrix(path: Path) -> tuple[ExpressionProfile, ...]:
    lines = open_text(path)
    header = next(lines).rstrip("\n").split("\t")
    patient_ids = tuple(tcga_patient_identifier(item) for item in header[1:])
    genes: list[str] = []
    columns: list[list[float]] = [[] for _ in patient_ids]
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != len(header):
            continue
        genes.append(fields[0].split("|")[0])
        for index, value in enumerate(fields[1:]):
            columns[index].append(parse_float(value, 0.0))
    gene_tuple = tuple(genes)
    return tuple(
        ExpressionProfile(identifier, gene_tuple, np.asarray(values, dtype=np.float32))
        for identifier, values in zip(patient_ids, columns, strict=True)
    )


def variance_ranked_genes(
    profiles: Sequence[ExpressionProfile],
    count: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    if len(profiles) == 0:
        raise ValueError("expression profiles cannot be empty")
    matrix = np.stack([profile.values for profile in profiles])
    variances = np.var(matrix, axis=0)
    selected = np.argsort(variances)[::-1][:count]
    genes = tuple(profiles[0].genes[index] for index in selected)
    return genes, matrix[:, selected].astype(np.float32)


def read_pathways(path: Path) -> tuple[PathwayDefinition, ...]:
    pathways: list[PathwayDefinition] = []
    for line in open_text(path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        pathways.append(PathwayDefinition(fields[0], fields[1], frozenset(fields[2:])))
    return tuple(pathways)


def pathway_membership(genes: Sequence[str], pathways: Sequence[PathwayDefinition]) -> np.ndarray:
    matrix = np.zeros((len(pathways), len(genes)), dtype=np.float32)
    for pathway_index, pathway in enumerate(pathways):
        for gene_index, gene in enumerate(genes):
            if gene in pathway.genes:
                matrix[pathway_index, gene_index] = 1.0
    denominator = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(denominator, 1.0)


def pathway_tokens(expression: np.ndarray, membership: np.ndarray) -> np.ndarray:
    return np.matmul(expression, membership.T).astype(np.float32)


def read_mimic_events(
    path: Path,
    patient_field: str,
    timestamp_field: str,
    category: str,
    code_field: str,
    value_field: str | None = None,
    unit_field: str | None = None,
) -> tuple[TimelineEvent, ...]:
    events: list[TimelineEvent] = []
    for row in parse_delimited(path, delimiter=","):
        timestamp = parse_date(row.get(timestamp_field))
        if timestamp is None:
            continue
        value = parse_float(row.get(value_field)) if value_field is not None else None
        unit = row.get(unit_field) if unit_field is not None else None
        events.append(
            TimelineEvent(
                row[patient_field],
                timestamp,
                category,
                row.get(code_field, ""),
                value,
                unit,
            )
        )
    return tuple(events)


def align_events_to_cycles(
    events: Iterable[TimelineEvent],
    cycles: Sequence[TreatmentCycle],
) -> dict[tuple[str, int], list[TimelineEvent]]:
    by_patient: dict[str, list[TimelineEvent]] = defaultdict(list)
    for event in events:
        by_patient[event.patient_id].append(event)
    aligned: dict[tuple[str, int], list[TimelineEvent]] = defaultdict(list)
    for cycle in cycles:
        patient_events = by_patient.get(cycle.patient_id, [])
        for event in patient_events:
            after_start = event.timestamp >= cycle.started
            before_end = cycle.ended is None or event.timestamp < cycle.ended
            if after_start and before_end:
                aligned[(cycle.patient_id, cycle.index)].append(event)
    for values in aligned.values():
        values.sort(key=lambda event: event.timestamp)
    return dict(aligned)


def temporal_feature_matrix(
    events: Sequence[TimelineEvent],
    vocabulary: Mapping[tuple[str, str], int],
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(events) == 0:
        return np.zeros((1, feature_count), dtype=np.float32), np.zeros(1, dtype=np.float32)
    origin = events[0].timestamp
    matrix = np.zeros((len(events), feature_count), dtype=np.float32)
    times = np.zeros(len(events), dtype=np.float32)
    for index, event in enumerate(events):
        feature = vocabulary.get((event.category, event.code))
        if feature is not None and feature < feature_count:
            matrix[index, feature] = 1.0 if event.numeric_value is None else event.numeric_value
        times[index] = (event.timestamp - origin).total_seconds() / 3600.0
    return matrix, times


def load_cbioportal_cases(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("cBioPortal case payload must be a list")
    cases: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        cases.append(dict(item))
    return tuple(cases)


def harmonize_cohort_identifiers(
    clinical: Sequence[ClinicalPatient],
    expression: Sequence[ExpressionProfile],
    mutations: Sequence[MutationCall],
) -> tuple[str, ...]:
    clinical_ids = {item.identifier for item in clinical}
    expression_ids = {item.patient_id for item in expression}
    mutation_ids = {item.patient_id for item in mutations}
    return tuple(sorted(clinical_ids & expression_ids & mutation_ids))


def recist_category(baseline: float, current: float, new_lesion: bool = False) -> str:
    if new_lesion:
        return "progressive_disease"
    if baseline <= 0.0:
        raise ValueError("baseline burden must be positive")
    change = (current - baseline) / baseline
    if current == 0.0:
        return "complete_response"
    if change <= -0.30:
        return "partial_response"
    if change >= 0.20 and current - baseline >= 5.0:
        return "progressive_disease"
    return "stable_disease"


def response_target(category: str) -> int:
    normalized = category.strip().lower().replace(" ", "_")
    if normalized in {"complete_response", "partial_response"}:
        return 1
    if normalized in {"stable_disease", "progressive_disease"}:
        return 0
    raise ValueError(f"unknown response category: {category}")


def ctcae_grade_target(grades: Sequence[int], threshold: int = 3) -> np.ndarray:
    return np.asarray([int(grade >= threshold) for grade in grades], dtype=np.float32)

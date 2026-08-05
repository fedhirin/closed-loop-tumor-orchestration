from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tumor_orchestration.records import DrugConstraintProfile


@dataclass(frozen=True)
class Atom:
    index: int
    atomic_number: int
    degree: int
    formal_charge: int
    aromatic: bool
    symbol: str


@dataclass(frozen=True)
class Bond:
    source: int
    target: int
    order: float
    aromatic: bool


@dataclass(frozen=True)
class MolecularGraph:
    identifier: str
    atoms: tuple[Atom, ...]
    bonds: tuple[Bond, ...]
    fingerprint: np.ndarray


@dataclass(frozen=True)
class DrugInteraction:
    first: str
    second: str
    severity: float
    mechanism: str


@dataclass(frozen=True)
class DrugTarget:
    drug: str
    target: str
    action: str
    confidence: float


@dataclass(frozen=True)
class TreatmentCandidate:
    identifier: str
    dose: float
    unit: str
    route: str
    line: int


def stable_hash(value: str, modulus: int) -> int:
    state = 2166136261
    for byte in value.encode("utf-8"):
        state ^= byte
        state = state * 16777619 & 0xFFFFFFFF
    return state % modulus


def token_fingerprint(tokens: Iterable[str], dimensions: int = 2048) -> np.ndarray:
    fingerprint = np.zeros(dimensions, dtype=np.float32)
    for token in tokens:
        fingerprint[stable_hash(token, dimensions)] = 1.0
    return fingerprint


def smiles_fragments(smiles: str, radius: int = 2) -> tuple[str, ...]:
    atoms = re.findall(r"Cl|Br|[A-Z][a-z]?|[cnosp]|\d+|\(|\)|=|#", smiles)
    fragments: list[str] = []
    for center in range(len(atoms)):
        for distance in range(radius + 1):
            start = max(0, center - distance)
            end = min(len(atoms), center + distance + 1)
            fragments.append("".join(atoms[start:end]))
    return tuple(fragments)


def morgan_like_fingerprint(smiles: str, dimensions: int = 2048, radius: int = 2) -> np.ndarray:
    return token_fingerprint(smiles_fragments(smiles, radius), dimensions)


def atom_feature(atom: Atom) -> np.ndarray:
    atomic_numbers = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    feature = np.zeros(32, dtype=np.float32)
    if atom.atomic_number in atomic_numbers:
        feature[atomic_numbers.index(atom.atomic_number)] = 1.0
    else:
        feature[10] = 1.0
    feature[11 + min(max(atom.degree, 0), 5)] = 1.0
    charge_index = min(max(atom.formal_charge + 3, 0), 6)
    feature[17 + charge_index] = 1.0
    feature[24] = float(atom.aromatic)
    feature[25] = atom.atomic_number / 100.0
    feature[26] = atom.degree / 8.0
    feature[27] = atom.formal_charge / 5.0
    feature[28 + stable_hash(atom.symbol, 4)] = 1.0
    return feature


def adjacency_matrix(graph: MolecularGraph, self_loops: bool = True) -> np.ndarray:
    size = len(graph.atoms)
    adjacency = np.zeros((size, size), dtype=np.float32)
    for bond in graph.bonds:
        adjacency[bond.source, bond.target] = bond.order
        adjacency[bond.target, bond.source] = bond.order
    if self_loops:
        np.fill_diagonal(adjacency, 1.0)
    return adjacency


def graph_features(graph: MolecularGraph, dimensions: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.zeros((len(graph.atoms), dimensions), dtype=np.float32)
    for index, atom in enumerate(graph.atoms):
        base = atom_feature(atom)
        nodes[index, : base.size] = base
        nodes[index, base.size :] = graph.fingerprint[base.size :]
    return nodes, adjacency_matrix(graph)


class DrugInteractionNetwork:
    def __init__(
        self,
        interactions: Iterable[DrugInteraction],
        targets: Iterable[DrugTarget],
        minimum_confidence: float = 0.7,
    ) -> None:
        self.interactions: dict[str, dict[str, DrugInteraction]] = defaultdict(dict)
        self.targets: dict[str, list[DrugTarget]] = defaultdict(list)
        for interaction in interactions:
            self.interactions[interaction.first][interaction.second] = interaction
            self.interactions[interaction.second][interaction.first] = interaction
        for target in targets:
            if target.confidence > minimum_confidence:
                self.targets[target.drug].append(target)

    def interaction_risk(self, candidate: str, active: Sequence[str]) -> float:
        risks = [
            self.interactions[candidate][other].severity
            for other in active
            if other in self.interactions.get(candidate, {})
        ]
        return max(risks, default=0.0)

    def target_overlap(self, first: str, second: str) -> float:
        first_targets = {item.target for item in self.targets.get(first, [])}
        second_targets = {item.target for item in self.targets.get(second, [])}
        union = first_targets | second_targets
        return len(first_targets & second_targets) / len(union) if union else 0.0

    def candidate_matrix(self, candidates: Sequence[str], active: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(candidates), len(candidates)), dtype=np.float32)
        for row, first in enumerate(candidates):
            matrix[row, row] = self.interaction_risk(first, active)
            for column, second in enumerate(candidates):
                if row != column:
                    matrix[row, column] = self.target_overlap(first, second)
        return matrix


class ConstraintBuilder:
    def __init__(self, action_names: Sequence[str], network: DrugInteractionNetwork) -> None:
        self.action_names = tuple(action_names)
        self.network = network

    def build(
        self,
        active_medications: Sequence[Sequence[str]],
        allergies: Sequence[set[str]],
        contraindications: Sequence[set[str]],
        comorbidity_risk: Tensor,
    ) -> DrugConstraintProfile:
        batch = len(active_medications)
        actions = len(self.action_names)
        interaction = torch.zeros(batch, actions, dtype=torch.float32)
        allergy = torch.zeros_like(interaction)
        contraindicated = torch.zeros_like(interaction, dtype=torch.bool)
        for patient in range(batch):
            for action, name in enumerate(self.action_names):
                interaction[patient, action] = self.network.interaction_risk(
                    name,
                    active_medications[patient],
                )
                allergy[patient, action] = float(name in allergies[patient])
                contraindicated[patient, action] = name in contraindications[patient]
        return DrugConstraintProfile(
            contraindicated,
            interaction,
            allergy,
            comorbidity_risk,
        )


def read_interactions(path: Path) -> tuple[DrugInteraction, ...]:
    interactions: list[DrugInteraction] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            interactions.append(
                DrugInteraction(
                    row["drug_a"],
                    row["drug_b"],
                    float(row["severity"]),
                    row.get("mechanism", ""),
                )
            )
    return tuple(interactions)


def read_targets(path: Path) -> tuple[DrugTarget, ...]:
    targets: list[DrugTarget] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            targets.append(
                DrugTarget(
                    row["drug"],
                    row["target"],
                    row.get("action", "unknown"),
                    float(row["confidence"]),
                )
            )
    return tuple(targets)


def pad_graphs(
    graphs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[Tensor, Tensor, Tensor]:
    if len(graphs) == 0:
        raise ValueError("graph sequence cannot be empty")
    maximum_nodes = max(nodes.shape[0] for nodes, _ in graphs)
    feature_count = graphs[0][0].shape[1]
    nodes = torch.zeros(len(graphs), maximum_nodes, feature_count)
    adjacency = torch.zeros(len(graphs), maximum_nodes, maximum_nodes)
    valid = torch.zeros(len(graphs), maximum_nodes, dtype=torch.bool)
    for index, (node_array, adjacency_array) in enumerate(graphs):
        count = node_array.shape[0]
        nodes[index, :count] = torch.from_numpy(node_array)
        adjacency[index, :count, :count] = torch.from_numpy(adjacency_array)
        valid[index, :count] = True
    return nodes, adjacency, valid

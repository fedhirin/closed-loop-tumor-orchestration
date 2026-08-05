from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from collections.abc import Iterable, Mapping, MutableMapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from tumor_orchestration.agent import ClosedLoopTumorAgent
from tumor_orchestration.metrics import MetricAccumulator
from tumor_orchestration.objectives import MultitaskObjective, ppo_objective
from tumor_orchestration.policy import RolloutBuffer, generalized_advantage_estimate
from tumor_orchestration.records import EpochReport, TargetBatch
from tumor_orchestration.schema import AgentConfig

LOGGER = logging.getLogger("tumor_orchestration")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if distributed_available() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if distributed_available() else 1


def is_primary() -> bool:
    return rank() == 0


def initialize_distributed(backend: str | None = None) -> tuple[int, int, int]:
    if distributed_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank(), world_size(), local_rank
    requested_world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_world > 1:
        selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        torch.distributed.init_process_group(backend=selected_backend, init_method="env://")
        return rank(), world_size(), local_rank
    return 0, 1, local_rank


def finalize_distributed() -> None:
    if distributed_available():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def reduce_mean(value: Tensor) -> Tensor:
    if not distributed_available():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result / world_size()


def gather_tensor(value: Tensor) -> Tensor:
    if not distributed_available():
        return value
    outputs = [torch.empty_like(value) for _ in range(world_size())]
    torch.distributed.all_gather(outputs, value)
    return torch.cat(outputs, dim=0)


def configure_logging(verbosity: int = logging.INFO) -> None:
    logging.basicConfig(
        level=verbosity,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def device_for_local_rank(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def precision_context(config: AgentConfig, device: torch.device) -> AbstractContextManager[None]:
    if device.type != "cuda" or config.training.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if config.training.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cosine_warmup_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def create_optimizer(model: nn.Module, config: AgentConfig, policy: bool = False) -> AdamW:
    learning_rate = (
        config.training.policy_learning_rate if policy else config.training.learning_rate
    )
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "normalization" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        (
            {"params": decay, "weight_decay": config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    config: AgentConfig,
) -> LambdaLR:
    total_steps = config.training.supervised_epochs * steps_per_epoch
    warmup_steps = config.training.warmup_epochs * steps_per_epoch
    return LambdaLR(
        optimizer,
        lambda step: cosine_warmup_multiplier(step, warmup_steps, total_steps),
    )


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class TrainingState:
    epoch: int
    global_step: int
    best_validation_loss: float
    patience_used: int
    seed: int
    phase: str


class CheckpointManager:
    def __init__(self, directory: Path, keep: int = 3) -> None:
        self.directory = directory
        self.keep = keep
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR | None,
        scaler: GradScaler | None,
        state: TrainingState,
        config: AgentConfig,
    ) -> Path:
        target = self.directory / f"{name}.pt"
        module = model.module if isinstance(model, DistributedDataParallel) else model
        payload: dict[str, Any] = {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": asdict(state),
            "config": asdict(config),
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        if torch.cuda.is_available():
            payload["cuda_random"] = torch.cuda.get_rng_state_all()
        atomic_torch_save(payload, target)
        self._prune()
        return target

    def restore(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LambdaLR | None = None,
        scaler: GradScaler | None = None,
        map_location: torch.device | str = "cpu",
    ) -> TrainingState:
        payload = torch.load(path, map_location=map_location)
        module = model.module if isinstance(model, DistributedDataParallel) else model
        module.load_state_dict(payload["model"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        random.setstate(payload["python_random"])
        np.random.set_state(payload["numpy_random"])
        torch.set_rng_state(payload["torch_random"])
        if torch.cuda.is_available() and "cuda_random" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_random"])
        state = payload["state"]
        return TrainingState(**state)

    def _prune(self) -> None:
        paths = sorted(self.directory.glob("*.pt"), key=lambda item: item.stat().st_mtime)
        for path in paths[: max(0, len(paths) - self.keep)]:
            path.unlink()


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


def move_to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def targets_from_batch(batch: Mapping[str, Tensor]) -> TargetBatch:
    return TargetBatch(
        batch["response"],
        batch["survival_time"],
        batch["survival_event"],
        batch["toxicity"],
        batch["segmentation"],
        batch["actions"],
        batch["advantages"],
        batch["old_log_probabilities"],
    )


class SupervisedTrainer:
    def __init__(
        self,
        model: ClosedLoopTumorAgent,
        config: AgentConfig,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.objective = MultitaskObjective(config.loss, config.training.ppo_clip)
        enabled = device.type == "cuda" and config.training.precision == "fp16"
        self.scaler = GradScaler(enabled=enabled)
        self.ema = ExponentialMovingAverage(model)
        self.global_step = 0

    def train_epoch(self, batches: Iterable[Mapping[str, Tensor]], epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        count = 0
        self.optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(batches):
            batch = move_to_device(raw_batch, self.device)
            with precision_context(self.config, self.device):
                output = forward_batch(self.model, batch)
                losses = self.objective(
                    output.predictions, targets_from_batch(batch), output.tokens
                )
                scaled_loss = losses.total / self.config.training.gradient_accumulation
            self.scaler.scale(scaled_loss).backward()
            boundary = (index + 1) % self.config.training.gradient_accumulation == 0
            if boundary:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1
            running_loss += float(reduce_mean(losses.total.detach()).item())
            count += 1
        return running_loss / max(count, 1)

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Mapping[str, Tensor]]) -> tuple[float, dict[str, float]]:
        self.model.eval()
        accumulator = MetricAccumulator.create()
        total_loss = 0.0
        count = 0
        for raw_batch in batches:
            batch = move_to_device(raw_batch, self.device)
            with precision_context(self.config, self.device):
                output = forward_batch(self.model, batch)
                targets = targets_from_batch(batch)
                losses = self.objective(output.predictions, targets, output.tokens)
            predictions = output.predictions
            accumulator.update(
                targets.response,
                torch.sigmoid(predictions.response_logits),
                targets.survival_time,
                targets.survival_event,
                predictions.survival_risk,
                targets.toxicity,
                torch.sigmoid(predictions.toxicity_logits),
                targets.segmentation,
                torch.sigmoid(predictions.segmentation_logits),
                targets.actions,
                predictions.policy_logits.argmax(dim=-1),
            )
            total_loss += float(reduce_mean(losses.total).item())
            count += 1
        return total_loss / max(count, 1), accumulator.compute()


def forward_batch(model: ClosedLoopTumorAgent, batch: Mapping[str, Tensor]) -> Any:
    cycles = []
    count = batch["ct"].shape[1]
    from tumor_orchestration.records import CycleBatch

    for cycle in range(count):
        cycles.append(
            CycleBatch(
                batch["ct"][:, cycle],
                batch["expression"][:, cycle],
                batch["mutations"][:, cycle],
                batch["ehr"][:, cycle],
                batch["ehr_times"][:, cycle],
                batch["drug_nodes"][:, cycle],
                batch["drug_adjacency"][:, cycle],
                batch["treatment"][:, cycle],
                batch["modality_mask"][:, cycle],
            )
        )
    elapsed = batch["ehr_times"].mean(dim=-1)
    return model(tuple(cycles), elapsed, batch.get("valid_cycles"))


class PolicyTrainer:
    def __init__(
        self,
        model: ClosedLoopTumorAgent,
        optimizer: Optimizer,
        clip_ratio: float,
        epochs_per_rollout: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.epochs_per_rollout = epochs_per_rollout
        self.minibatch_size = minibatch_size

    def update(self, buffer: RolloutBuffer, bootstrap_value: Tensor) -> dict[str, float]:
        states, actions, old_logs, rewards, values, dones = buffer.stack()
        values_with_bootstrap = torch.cat((values, bootstrap_value.unsqueeze(1)), dim=1)
        advantages, returns = generalized_advantage_estimate(rewards, values_with_bootstrap, dones)
        states = states.flatten(0, 1)
        actions = actions.flatten(0, 1)
        old_logs = old_logs.flatten(0, 1)
        advantages = advantages.flatten(0, 1)
        returns = returns.flatten(0, 1)
        indices = torch.arange(states.shape[0], device=states.device)
        totals: MutableMapping[str, float] = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.epochs_per_rollout):
            permutation = indices[torch.randperm(indices.numel(), device=indices.device)]
            for start in range(0, indices.numel(), self.minibatch_size):
                selected = permutation[start : start + self.minibatch_size]
                output = self.model.orchestrator.policy(states[selected])
                loss = ppo_objective(
                    output.logits,
                    output.values,
                    actions[selected],
                    advantages[selected],
                    returns[selected],
                    old_logs[selected],
                    self.clip_ratio,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.orchestrator.parameters(), 1.0)
                self.optimizer.step()
                totals["policy"] += float(loss.policy.detach().item())
                totals["value"] += float(loss.value.detach().item())
                totals["entropy"] += float(loss.entropy.detach().item())
                updates += 1
        buffer.clear()
        return {name: value / max(updates, 1) for name, value in totals.items()}


class ReportWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, report: EpochReport) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(report), sort_keys=True) + "\n")


class EarlyStopping:
    def __init__(self, patience: int, minimum_delta: float = 0.0) -> None:
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best = float("inf")
        self.used = 0

    def update(self, value: float) -> bool:
        if value < self.best - self.minimum_delta:
            self.best = value
            self.used = 0
            return False
        self.used += 1
        return self.used >= self.patience


class WallClock:
    def __init__(self) -> None:
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

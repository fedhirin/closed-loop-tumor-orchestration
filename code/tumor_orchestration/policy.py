from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from tumor_orchestration.records import DrugConstraintProfile, SafetyDecision


class PolicyOutput(NamedTuple):
    logits: Tensor
    values: Tensor
    toxicity: Tensor


class TreatmentPolicy(nn.Module):
    def __init__(self, dimension: int, actions: int, dropout: float) -> None:
        super().__init__()
        self.actions = actions
        self.backbone = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(dimension, actions)
        self.value_head = nn.Linear(dimension, 1)
        self.toxicity_head = nn.Linear(dimension, actions)

    def forward(self, context: Tensor) -> PolicyOutput:
        hidden = self.backbone(context)
        return PolicyOutput(
            self.action_head(hidden),
            self.value_head(hidden).squeeze(-1),
            self.toxicity_head(hidden),
        )

    def distribution(self, context: Tensor, safe_mask: Tensor | None = None) -> Categorical:
        logits = self(context).logits
        if safe_mask is not None:
            logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)


class PharmacologicalGuardrails(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, toxicity_logits: Tensor, profile: DrugConstraintProfile) -> Tensor:
        toxicity_probability = torch.sigmoid(toxicity_logits)
        safe = profile.combined_mask(toxicity_probability, self.threshold)
        no_safe_action = ~safe.any(dim=-1)
        if no_safe_action.any():
            least_risky = toxicity_probability.argmin(dim=-1)
            rows = torch.arange(safe.shape[0], device=safe.device)
            safe = safe.clone()
            safe[rows[no_safe_action], least_risky[no_safe_action]] = True
        return safe


class MonteCarloEscalation(nn.Module):
    def __init__(self, samples: int) -> None:
        super().__init__()
        self.samples = samples

    def entropy(self, probabilities: Tensor) -> Tensor:
        bounded = probabilities.clamp_min(torch.finfo(probabilities.dtype).eps)
        return -(bounded * bounded.log()).sum(dim=-1)

    def forward(
        self,
        policy: TreatmentPolicy,
        context: Tensor,
        safe_mask: Tensor,
        threshold: Tensor,
    ) -> SafetyDecision:
        previous_mode = policy.training
        policy.train(True)
        probabilities: list[Tensor] = []
        for _ in range(self.samples):
            logits = policy(context).logits.masked_fill(
                ~safe_mask,
                torch.finfo(context.dtype).min,
            )
            probabilities.append(torch.softmax(logits, dim=-1))
        policy.train(previous_mode)
        stacked = torch.stack(probabilities, dim=0)
        uncertainty = self.entropy(stacked).mean(dim=0)
        mean_probability = stacked.mean(dim=0)
        action = mean_probability.argmax(dim=-1)
        escalated = uncertainty > threshold
        return SafetyDecision(action, uncertainty, escalated, safe_mask, mean_probability)


class SafeOrchestrator(nn.Module):
    def __init__(
        self,
        dimension: int,
        actions: int,
        dropout: float,
        toxicity_threshold: float,
        monte_carlo_samples: int,
    ) -> None:
        super().__init__()
        self.policy = TreatmentPolicy(dimension, actions, dropout)
        self.guardrails = PharmacologicalGuardrails(toxicity_threshold)
        self.escalation = MonteCarloEscalation(monte_carlo_samples)
        self.register_buffer("escalation_threshold", torch.tensor(float("inf")))

    def calibrate(self, entropies: Tensor, percentile: float) -> Tensor:
        quantile = torch.quantile(entropies.float(), percentile / 100.0)
        self.escalation_threshold.copy_(quantile)
        return quantile

    def forward(self, context: Tensor) -> PolicyOutput:
        return self.policy(context)

    def decide(self, context: Tensor, profile: DrugConstraintProfile) -> SafetyDecision:
        policy_output = self.policy(context)
        safe_mask = self.guardrails(policy_output.toxicity, profile)
        threshold = self.escalation_threshold.expand(context.shape[0])
        return self.escalation(self.policy, context, safe_mask, threshold)


def treatment_reward(
    efficacy: Tensor,
    toxicity: Tensor,
    violated: Tensor,
    efficacy_weight: float,
    toxicity_weight: float,
    violation_penalty: float,
) -> Tensor:
    return (
        efficacy_weight * efficacy
        - toxicity_weight * toxicity
        - violation_penalty * violated.float()
    )


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    discount: float = 0.99,
    trace_decay: float = 0.95,
) -> tuple[Tensor, Tensor]:
    if rewards.shape != dones.shape:
        raise ValueError("rewards and terminal flags must have equal shape")
    if values.shape[-1] != rewards.shape[-1] + 1:
        raise ValueError("values must contain one bootstrap element")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[..., 0])
    for index in range(rewards.shape[-1] - 1, -1, -1):
        continuation = 1.0 - dones[..., index].float()
        delta = (
            rewards[..., index]
            + discount * values[..., index + 1] * continuation
            - values[..., index]
        )
        running = delta + discount * trace_decay * continuation * running
        advantages[..., index] = running
    returns = advantages + values[..., :-1]
    return advantages, returns


class RolloutBuffer:
    def __init__(self) -> None:
        self.states: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.log_probabilities: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.values: list[Tensor] = []
        self.dones: list[Tensor] = []

    def append(
        self,
        state: Tensor,
        action: Tensor,
        log_probability: Tensor,
        reward: Tensor,
        value: Tensor,
        done: Tensor,
    ) -> None:
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probabilities.append(log_probability.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.dones.append(done.detach())

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def stack(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if len(self) == 0:
            raise ValueError("rollout buffer is empty")
        return (
            torch.stack(self.states, dim=1),
            torch.stack(self.actions, dim=1),
            torch.stack(self.log_probabilities, dim=1),
            torch.stack(self.rewards, dim=1),
            torch.stack(self.values, dim=1),
            torch.stack(self.dones, dim=1),
        )

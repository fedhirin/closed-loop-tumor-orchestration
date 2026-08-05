from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tumor_orchestration.records import AgentPredictions, LossBreakdown, ModalityTokens, TargetBatch
from tumor_orchestration.schema import LossConfig


class PPOLoss(NamedTuple):
    policy: Tensor
    value: Tensor
    entropy: Tensor
    total: Tensor


def binary_response_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits.reshape_as(targets), targets.float())


def cox_partial_likelihood(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order].float()
    log_cumulative_hazard = torch.logcumsumexp(ordered_risk, dim=0)
    contributions = (ordered_risk - log_cumulative_hazard) * ordered_event
    return -contributions.sum() / ordered_event.sum().clamp_min(1.0)


def multilabel_toxicity_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def soft_dice_loss(logits: Tensor, targets: Tensor, smoothing: float = 1.0) -> Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(1)
    flattened_targets = targets.float().flatten(1)
    intersection = (probabilities * flattened_targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + flattened_targets.sum(dim=1)
    score = (2.0 * intersection + smoothing) / (denominator + smoothing)
    return 1.0 - score.mean()


def symmetric_contrastive_loss(left: Tensor, right: Tensor, temperature: float = 0.07) -> Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = torch.matmul(left, right.transpose(0, 1)) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    forward = F.cross_entropy(logits, labels)
    backward = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (forward + backward)


def modality_alignment_loss(tokens: ModalityTokens, temperature: float = 0.07) -> Tensor:
    pooled = [item.mean(dim=1) for item in tokens]
    losses: list[Tensor] = []
    for left in range(len(pooled)):
        for right in range(left + 1, len(pooled)):
            losses.append(symmetric_contrastive_loss(pooled[left], pooled[right], temperature))
    return torch.stack(losses).mean()


def clipped_policy_loss(
    logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    old_log_probabilities: Tensor,
    clip_ratio: float,
) -> Tensor:
    distribution = torch.distributions.Categorical(logits=logits)
    log_probabilities = distribution.log_prob(actions)
    ratio = torch.exp(log_probabilities - old_log_probabilities)
    normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    unclipped = ratio * normalized
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * normalized
    return -torch.minimum(unclipped, clipped).mean()


def ppo_objective(
    logits: Tensor,
    values: Tensor,
    actions: Tensor,
    advantages: Tensor,
    returns: Tensor,
    old_log_probabilities: Tensor,
    clip_ratio: float,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> PPOLoss:
    distribution = torch.distributions.Categorical(logits=logits)
    log_probabilities = distribution.log_prob(actions)
    ratio = torch.exp(log_probabilities - old_log_probabilities)
    normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    policy = -torch.minimum(
        ratio * normalized,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * normalized,
    ).mean()
    value = F.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    total = policy + value_weight * value - entropy_weight * entropy
    return PPOLoss(policy, value, entropy, total)


class MultitaskObjective(nn.Module):
    def __init__(self, weights: LossConfig, ppo_clip: float) -> None:
        super().__init__()
        self.weights = weights
        self.ppo_clip = ppo_clip

    def forward(
        self,
        predictions: AgentPredictions,
        targets: TargetBatch,
        tokens: ModalityTokens,
    ) -> LossBreakdown:
        response = binary_response_loss(predictions.response_logits, targets.response)
        survival = cox_partial_likelihood(
            predictions.survival_risk,
            targets.survival_time,
            targets.survival_event,
        )
        toxicity = multilabel_toxicity_loss(predictions.toxicity_logits, targets.toxicity)
        segmentation = soft_dice_loss(predictions.segmentation_logits, targets.segmentation)
        policy = clipped_policy_loss(
            predictions.policy_logits,
            targets.actions,
            targets.advantages,
            targets.old_log_probabilities,
            self.ppo_clip,
        )
        alignment = modality_alignment_loss(tokens)
        total = (
            response
            + self.weights.survival * survival
            + self.weights.toxicity * toxicity
            + self.weights.segmentation * segmentation
            + self.weights.policy * policy
            + self.weights.alignment * alignment
        )
        return LossBreakdown(
            total,
            response,
            survival,
            toxicity,
            segmentation,
            policy,
            alignment,
        )

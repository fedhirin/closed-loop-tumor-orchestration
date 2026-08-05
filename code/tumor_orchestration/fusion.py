from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tumor_orchestration.records import ModalityTokens


class FusionOutput(NamedTuple):
    fused: Tensor
    modalities: ModalityTokens
    gates: Tensor


class TreatmentConditioner(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.condition = nn.Linear(dimension, dimension, bias=False)
        self.queries = nn.ModuleDict(
            {
                name: nn.Linear(dimension, dimension, bias=False)
                for name in ("imaging", "genomics", "ehr", "pharmacology")
            }
        )

    def forward(self, tokens: ModalityTokens, treatment: Tensor) -> ModalityTokens:
        condition = self.condition(treatment).unsqueeze(1)
        return ModalityTokens(
            self.queries["imaging"](tokens.imaging + condition),
            self.queries["genomics"](tokens.genomics + condition),
            self.queries["ehr"](tokens.ehr + condition),
            self.queries["pharmacology"](tokens.pharmacology + condition),
        )


class CrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_normalization = nn.LayerNorm(dimension)
        self.context_normalization = nn.LayerNorm(dimension)
        self.output_normalization = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_query = self.query_normalization(query)
        normalized_context = self.context_normalization(context)
        attended, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        return self.output_normalization(query + self.dropout(attended))


class FeedForward(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.input_normalization = nn.LayerNorm(dimension)
        self.input_projection = nn.Linear(dimension, dimension * 4)
        self.output_projection = nn.Linear(dimension * 4, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.input_projection(self.input_normalization(inputs))
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        return inputs + self.dropout(self.output_projection(hidden))


class ModalityGate(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(dimension, 1)
                for name in ("imaging", "genomics", "ehr", "pharmacology")
            }
        )

    def _gate(self, name: str, tokens: Tensor, availability: Tensor) -> tuple[Tensor, Tensor]:
        score = self.projections[name](tokens.mean(dim=1)).squeeze(-1)
        gate = torch.sigmoid(score) * availability
        return tokens * gate[:, None, None], gate

    def forward(
        self, tokens: ModalityTokens, availability: Tensor
    ) -> tuple[ModalityTokens, Tensor]:
        imaging, image_gate = self._gate("imaging", tokens.imaging, availability[:, 0])
        genomics, genomic_gate = self._gate("genomics", tokens.genomics, availability[:, 1])
        ehr, ehr_gate = self._gate("ehr", tokens.ehr, availability[:, 2])
        pharmacology, pharmacology_gate = self._gate(
            "pharmacology",
            tokens.pharmacology,
            availability[:, 3],
        )
        gates = torch.stack((image_gate, genomic_gate, ehr_gate, pharmacology_gate), dim=-1)
        return ModalityTokens(imaging, genomics, ehr, pharmacology), gates


class CTIGALayer(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.conditioner = TreatmentConditioner(dimension)
        self.image_from_genomics = CrossAttention(dimension, heads, dropout)
        self.genomics_from_image = CrossAttention(dimension, heads, dropout)
        self.ehr_from_pharmacology = CrossAttention(dimension, heads, dropout)
        self.pharmacology_from_ehr = CrossAttention(dimension, heads, dropout)
        self.clinical_from_biology = CrossAttention(dimension, heads, dropout)
        self.biology_from_clinical = CrossAttention(dimension, heads, dropout)
        self.gate = ModalityGate(dimension)
        self.feed_forward = FeedForward(dimension, dropout)

    def _resize(self, source: Tensor, length: int) -> Tensor:
        if source.shape[1] == length:
            return source
        return F.adaptive_avg_pool1d(source.transpose(1, 2), length).transpose(1, 2)

    def forward(
        self,
        tokens: ModalityTokens,
        treatment: Tensor,
        availability: Tensor,
    ) -> FusionOutput:
        queries = self.conditioner(tokens, treatment)
        image = self.image_from_genomics(queries.imaging, tokens.genomics)
        genomics = self.genomics_from_image(queries.genomics, tokens.imaging)
        ehr = self.ehr_from_pharmacology(queries.ehr, tokens.pharmacology)
        pharmacology = self.pharmacology_from_ehr(queries.pharmacology, tokens.ehr)
        gated, gates = self.gate(ModalityTokens(image, genomics, ehr, pharmacology), availability)
        biology = torch.cat((gated.imaging, gated.genomics), dim=1)
        clinical = torch.cat((gated.ehr, gated.pharmacology), dim=1)
        biology_fused = self.biology_from_clinical(biology, clinical)
        clinical_fused = self.clinical_from_biology(clinical, biology)
        resized_clinical = self._resize(clinical_fused, biology_fused.shape[1])
        fused = self.feed_forward(biology_fused + resized_clinical)
        return FusionOutput(fused, gated, gates)


class CTIGA(nn.Module):
    def __init__(self, dimension: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(CTIGALayer(dimension, heads, dropout) for _ in range(layers))
        self.redistribution = nn.ModuleList(
            nn.ModuleDict(
                {
                    name: nn.Linear(dimension, dimension)
                    for name in ("imaging", "genomics", "ehr", "pharmacology")
                }
            )
            for _ in range(max(layers - 1, 0))
        )
        self.output_normalization = nn.LayerNorm(dimension)

    def _carry(self, tokens: ModalityTokens, projection: nn.ModuleDict) -> ModalityTokens:
        return ModalityTokens(
            projection["imaging"](tokens.imaging),
            projection["genomics"](tokens.genomics),
            projection["ehr"](tokens.ehr),
            projection["pharmacology"](tokens.pharmacology),
        )

    def forward(
        self,
        tokens: ModalityTokens,
        treatment: Tensor,
        availability: Tensor,
    ) -> FusionOutput:
        current = tokens
        gate_history: list[Tensor] = []
        fused = tokens.imaging
        for index, layer in enumerate(self.layers):
            result = layer(current, treatment, availability)
            fused = result.fused
            current = result.modalities
            gate_history.append(result.gates)
            if index < len(self.redistribution):
                current = self._carry(current, self.redistribution[index])
        gates = torch.stack(gate_history, dim=1)
        return FusionOutput(self.output_normalization(fused), current, gates)


class WeightedTokenPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, 1),
        )
        self.output = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension))

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return self.output(pooled)

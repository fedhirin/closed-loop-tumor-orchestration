from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tumor_orchestration.schema import DataConfig, ModelConfig


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualVolumeBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution_a = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.normalization_a = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.convolution_b = nn.Conv3d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization_b = nn.GroupNorm(_group_count(output_channels), output_channels)
        if input_channels == output_channels and stride == 1:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    input_channels, output_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.gelu(self.normalization_a(self.convolution_a(inputs)))
        hidden = self.normalization_b(self.convolution_b(hidden))
        return F.gelu(hidden + self.shortcut(inputs))


class ResidualVolumeStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, blocks: int, stride: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [ResidualVolumeBlock(input_channels, output_channels, stride)]
        layers.extend(
            ResidualVolumeBlock(output_channels, output_channels) for _ in range(blocks - 1)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class RECISTPool(nn.Module):
    def __init__(self, tokens: int) -> None:
        super().__init__()
        self.tokens = tokens
        depth = max(1, round(tokens ** (1.0 / 3.0)))
        height = max(1, round(math.sqrt(tokens / depth)))
        width = max(1, math.ceil(tokens / (depth * height)))
        self.output_size = (depth, height, width)

    def forward(self, features: Tensor, regions: Tensor | None = None) -> Tensor:
        if regions is not None:
            if regions.ndim != 5 or regions.shape[1] != 1:
                raise ValueError("regions must have shape batch by one by depth by height by width")
            mask = F.interpolate(regions.float(), size=features.shape[-3:], mode="nearest")
            features = features * mask
        pooled = F.adaptive_avg_pool3d(features, self.output_size)
        tokens = pooled.flatten(2).transpose(1, 2)
        if tokens.shape[1] < self.tokens:
            pad = self.tokens - tokens.shape[1]
            tokens = F.pad(tokens, (0, 0, 0, pad))
        return tokens[:, : self.tokens]


class CTVolumeEncoder(nn.Module):
    def __init__(self, dimension: int, tokens: int) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 512)
        self.stem = nn.Sequential(
            nn.Conv3d(1, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.stages = nn.Sequential(
            ResidualVolumeStage(widths[0], widths[1], 3, 1),
            ResidualVolumeStage(widths[1], widths[2], 4, 2),
            ResidualVolumeStage(widths[2], widths[3], 6, 2),
            ResidualVolumeStage(widths[3], widths[4], 3, 2),
        )
        self.pool = RECISTPool(tokens)
        self.projection = nn.Linear(widths[-1], dimension)
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, volumes: Tensor, regions: Tensor | None = None) -> Tensor:
        if volumes.ndim == 4:
            volumes = volumes.unsqueeze(1)
        hidden = self.stages(self.stem(volumes))
        return self.normalization(self.projection(self.pool(hidden, regions)))


class PathwayTokenizer(nn.Module):
    def __init__(
        self,
        expression_features: int,
        mutation_features: int,
        pathways: int,
        dimension: int,
    ) -> None:
        super().__init__()
        total_features = expression_features + mutation_features
        self.pathways = pathways
        self.feature_weight = nn.Parameter(torch.empty(pathways, total_features))
        self.feature_bias = nn.Parameter(torch.zeros(pathways))
        self.value_projection = nn.Linear(1, dimension)
        self.pathway_embedding = nn.Parameter(torch.empty(1, pathways, dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.normal_(self.pathway_embedding, std=0.02)

    def forward(self, expression: Tensor, mutations: Tensor) -> Tensor:
        features = torch.cat((expression, mutations), dim=-1)
        pathway_values = F.linear(features, self.feature_weight, self.feature_bias)
        return self.value_projection(pathway_values.unsqueeze(-1)) + self.pathway_embedding


class GenomicPathwayEncoder(nn.Module):
    def __init__(self, data: DataConfig, model: ModelConfig) -> None:
        super().__init__()
        self.tokenizer = PathwayTokenizer(
            data.gene_expression_features,
            data.mutation_features,
            data.genomic_tokens,
            model.dimension,
        )
        block = nn.TransformerEncoderLayer(
            d_model=model.dimension,
            nhead=model.heads,
            dim_feedforward=model.dimension * 4,
            dropout=model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, model.genomic_layers)
        self.normalization = nn.LayerNorm(model.dimension)

    def forward(self, expression: Tensor, mutations: Tensor) -> Tensor:
        return self.normalization(self.encoder(self.tokenizer(expression, mutations)))


class DilatedTemporalBlock(nn.Module):
    def __init__(self, dimension: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.convolution_a = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.convolution_b = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.normalization_a = nn.LayerNorm(dimension)
        self.normalization_b = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.convolution_a(inputs.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(F.gelu(self.normalization_a(hidden)))
        hidden = self.convolution_b(hidden.transpose(1, 2)).transpose(1, 2)
        return self.normalization_b(inputs + self.dropout(hidden))


class StructuredEHREncoder(nn.Module):
    def __init__(self, features: int, dimension: int, tokens: int, dropout: float) -> None:
        super().__init__()
        self.tokens = tokens
        self.feature_projection = nn.Linear(features, dimension)
        self.time_projection = nn.Sequential(
            nn.Linear(4, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.blocks = nn.ModuleList(
            [DilatedTemporalBlock(dimension, dilation, dropout) for dilation in (1, 2, 4, 8)]
        )
        self.attention = nn.Linear(dimension, 1)
        self.normalization = nn.LayerNorm(dimension)

    def time_features(self, times: Tensor) -> Tensor:
        scaled = times / 24.0
        return torch.stack(
            (scaled, torch.log1p(scaled.abs()), torch.sin(scaled), torch.cos(scaled)),
            dim=-1,
        )

    def forward(self, events: Tensor, times: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden = self.feature_projection(events) + self.time_projection(self.time_features(times))
        for block in self.blocks:
            hidden = block(hidden)
        if mask is not None:
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        if hidden.shape[1] != self.tokens:
            hidden = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.tokens).transpose(1, 2)
        weights = torch.softmax(self.attention(hidden), dim=1)
        context = torch.sum(hidden * weights, dim=1, keepdim=True)
        return self.normalization(hidden + context)


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(input_dimension, output_dimension, bias=False)
        self.key = nn.Linear(input_dimension, output_dimension, bias=False)
        self.value = nn.Linear(input_dimension, output_dimension, bias=False)
        self.output = nn.Linear(output_dimension, output_dimension)
        self.normalization = nn.LayerNorm(output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.scale = output_dimension**-0.5

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        identity = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=torch.bool)
        connected = adjacency.bool() | identity.unsqueeze(0)
        scores = scores.masked_fill(~connected, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        hidden = self.output(torch.matmul(weights, value))
        residual = nodes if nodes.shape[-1] == hidden.shape[-1] else value
        return self.normalization(residual + self.dropout(hidden))


class PharmacologyGraphEncoder(nn.Module):
    def __init__(self, data: DataConfig, model: ModelConfig) -> None:
        super().__init__()
        self.tokens = data.pharmacology_tokens
        self.input_projection = nn.Linear(data.drug_fingerprint_features, model.dimension)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(model.dimension, model.dimension, model.dropout)
            for _ in range(model.pharmacology_layers)
        )
        self.pool_score = nn.Linear(model.dimension, self.tokens)
        self.normalization = nn.LayerNorm(model.dimension)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        hidden = self.input_projection(nodes)
        for layer in self.layers:
            hidden = F.gelu(layer(hidden, adjacency))
        assignments = torch.softmax(self.pool_score(hidden).transpose(1, 2), dim=-1)
        pooled = torch.matmul(assignments, hidden)
        return self.normalization(pooled)

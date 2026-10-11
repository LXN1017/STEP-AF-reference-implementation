"""Hierarchical segment means, consistency fusion and a one-block Transformer."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    days: int = 96
    cycle_days: int = 8
    scales: tuple = (2, 4, 8)
    input_features: int = 10
    dimension: int = 32
    heads: int = 2
    feedforward: int = 64
    dropout: float = 0.3
    fusion: str = "consistency"
    stage_indices: tuple | None = None
    epsilon: float = 1e-8

    def validate(self):
        if self.days % self.cycle_days or len(set(self.scales)) != len(self.scales):
            raise ValueError("Cycles must partition the observation period; scales must be unique.")
        if any(s <= 0 or self.cycle_days % s for s in self.scales):
            raise ValueError("Every internal scale must divide the local cycle.")
        if len(self.scales) < 2 or self.dimension % self.heads:
            raise ValueError("Fusion requires >=2 scales; attention dimension must divide by heads.")
        if self.stage_indices is not None:
            if not self.stage_indices or any(not 0 <= k < self.days // self.cycle_days
                                              for k in self.stage_indices):
                raise ValueError("Stage indices are empty or out of range.")
            if tuple(sorted(set(self.stage_indices))) != tuple(self.stage_indices):
                raise ValueError("Retained stages must be unique and chronological.")
        if self.fusion not in {"consistency", "concat", "attention"}:
            raise ValueError("Unsupported fusion.")


class STEP_AF(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        c = self.config
        self.encoders = nn.ModuleList([
            nn.Linear(c.cycle_days // s * c.input_features, c.dimension) for s in c.scales])
        self.raw_a = nn.Parameter(torch.tensor(math.log(math.expm1(1. - c.epsilon))))
        self.b = nn.Parameter(torch.tensor(0.))
        if c.fusion == "concat":
            self.concat = nn.Linear(len(c.scales) * c.dimension, c.dimension)
        if c.fusion == "attention":
            self.attention = nn.Linear(c.dimension, 1)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=c.dimension, nhead=c.heads, dim_feedforward=c.feedforward,
            dropout=c.dropout, activation="gelu", batch_first=True, norm_first=False)
        self.classifier = nn.Sequential(nn.Linear(c.dimension, 16), nn.ReLU(),
                                        nn.Dropout(c.dropout), nn.Linear(16, 2))
        # Sinusoidal positional encoding is a documented reference implementation choice.
        position = torch.arange(c.days // c.cycle_days, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, c.dimension, 2, dtype=torch.float32)
                              * (-math.log(10000.) / c.dimension))
        pe = torch.zeros(c.days // c.cycle_days, c.dimension)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency[:pe[:, 1::2].shape[1]])
        self.register_buffer("positions", pe)

    def segment_means(self, x, scale):
        c = self.config
        n = x.shape[0]
        if x.shape[1:] != (c.days, c.input_features):
            raise ValueError("Daily input shape must agree with the model configuration.")
        return x.reshape(n, c.days // c.cycle_days, c.cycle_days // scale,
                         scale, c.input_features).mean(dim=3)

    def local_representation(self, x):
        c = self.config
        z = torch.stack([F.relu(encoder(self.segment_means(x, s).flatten(2)))
                         for s, encoder in zip(c.scales, self.encoders)], dim=2)
        if c.fusion == "concat":
            return F.relu(self.concat(z.flatten(2))), None
        if c.fusion == "attention":
            weights = self.attention(z).squeeze(-1).softmax(dim=-1)
            return (weights.unsqueeze(-1) * z).sum(dim=2), {"weights": weights}
        dot = torch.einsum("bksd,bkrd->bksr", z, z)
        norms = z.norm(dim=-1)
        cosine = dot / (norms.unsqueeze(-1) * norms.unsqueeze(-2) + c.epsilon)
        nscale = len(c.scales)
        off_diagonal = ~torch.eye(nscale, dtype=torch.bool, device=x.device)
        agreement = cosine.masked_fill(~off_diagonal, 0).sum(dim=-1) / (nscale - 1)
        weights = agreement.softmax(dim=-1)
        unique_pairs = torch.triu(torch.ones(nscale, nscale, dtype=torch.bool,
                                             device=x.device), diagonal=1)
        consistency = ((1 + cosine[:, :, unique_pairs]) / 2).mean(dim=-1)
        slope = F.softplus(self.raw_a) + c.epsilon
        gate = torch.sigmoid(slope * consistency + self.b)
        fused = gate.unsqueeze(-1) * (weights.unsqueeze(-1) * z).sum(dim=2)
        return fused, {"weights": weights, "gate": gate, "slope": slope,
                       "consistency": consistency}

    def forward(self, x, stage_permutation=None):
        stages, _ = self.local_representation(x)
        c = self.config
        kept = list(c.stage_indices) if c.stage_indices is not None \
            else list(range(c.days // c.cycle_days))
        stages = stages[:, kept]
        positions = self.positions[kept]
        if stage_permutation is not None:
            if stage_permutation.shape != stages.shape[:2]:
                raise ValueError("One complete permutation is required per patient.")
            expected = torch.arange(len(kept), device=stages.device).expand(stages.shape[0], -1)
            if not torch.equal(stage_permutation.sort(dim=-1).values, expected):
                raise ValueError("Permutation must contain every retained stage exactly once.")
            stages = stages.gather(1, stage_permutation.unsqueeze(-1).expand(-1, -1, c.dimension))
            # Slot positions, not source-stage positions: no original order is revealed.
            positions = self.positions[:len(kept)]
        encoded = self.transformer(stages + positions.unsqueeze(0))
        return self.classifier(encoded.mean(dim=1))

    def specification(self):
        return asdict(self.config)

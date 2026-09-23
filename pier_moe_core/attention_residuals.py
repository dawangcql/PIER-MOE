from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def _normalize_query(query: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if query.dim() == 1:
        denom = torch.rsqrt(query.float().pow(2).mean().clamp_min(eps))
    elif query.dim() == 2:
        denom = torch.rsqrt(query.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(eps))
    else:
        raise ValueError(f"query must have shape [D] or [Q, D], got {tuple(query.shape)}")
    return (query.float() * denom).to(query.dtype)


class AttentionResidualOperator(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.key_norm = RMSNorm(hidden_dim, eps=eps)
        self.logit_scale = hidden_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        values: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if values.dim() != 3:
            raise ValueError(f"values must have shape [B, S, D], got {tuple(values.shape)}")
        if query.dim() != 1:
            raise ValueError(f"query must have shape [D], got {tuple(query.shape)}")
        query = _normalize_query(query, eps=self.key_norm.eps)
        keys = self.key_norm(values)
        logits = torch.einsum("d,bsd->bs", query, keys) * self.logit_scale
        weights = torch.softmax(logits, dim=1)
        output = torch.einsum("bs,bsd->bd", weights, values)
        if return_weights:
            return output, weights
        return output


class SingleQueryAttentionResidual(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mode: str = "block",
        num_blocks: int = 4,
        preserve_first_source: bool = True,
        zero_init: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if mode not in {"full", "block"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.num_blocks = max(1, num_blocks)
        self.preserve_first_source = preserve_first_source
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        self.attn = AttentionResidualOperator(hidden_dim, eps=eps)
        if not zero_init:
            nn.init.normal_(self.query, mean=0.0, std=0.02)

    def _compress_blocks(self, values: torch.Tensor) -> torch.Tensor:
        if self.mode == "full" or values.size(1) <= 2:
            return values
        if self.preserve_first_source:
            prefix = values[:, :1, :]
            remainder = values[:, 1:, :]
        else:
            prefix = values[:, :0, :]
            remainder = values
        if remainder.size(1) == 0:
            return prefix
        num_blocks = min(self.num_blocks, remainder.size(1))
        block_size = int(math.ceil(remainder.size(1) / num_blocks))
        blocks = []
        for start in range(0, remainder.size(1), block_size):
            end = min(start + block_size, remainder.size(1))
            blocks.append(remainder[:, start:end, :].sum(dim=1, keepdim=True))
        return torch.cat([prefix] + blocks, dim=1)

    def forward(self, values: torch.Tensor, return_weights: bool = False):
        compressed = self._compress_blocks(values)
        output, weights = self.attn(self.query, compressed, return_weights=True)
        if return_weights:
            return output, weights
        return output


class ProjectedAttentionResidualFusion(nn.Module):
    """Project heterogeneous sources and select them with a single AttnRes query."""

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_dim: int,
        dropout: float = 0.1,
        mode: str = "block",
        num_blocks: int = 4,
        zero_init: bool = True,
        stable_residual: bool = True,
        residual_init_alpha: float = 0.05,
    ):
        super().__init__()
        self.proj_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for dim in input_dims
            ]
        )
        self.stable_residual = stable_residual
        self.selector = SingleQueryAttentionResidual(
            hidden_dim=hidden_dim,
            mode=mode,
            num_blocks=num_blocks,
            zero_init=zero_init,
        )
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if stable_residual:
            residual_init_alpha = min(max(residual_init_alpha, 1e-4), 1.0 - 1e-4)
            self.residual_logit = nn.Parameter(torch.tensor(math.log(residual_init_alpha / (1.0 - residual_init_alpha))))
            self.base_norm = nn.LayerNorm(hidden_dim)
        else:
            self.register_parameter("residual_logit", None)
            self.base_norm = nn.Identity()

    def forward(self, features_list: Sequence[torch.Tensor], return_weights: bool = False):
        if len(features_list) != len(self.proj_layers):
            raise ValueError(f"Expected {len(self.proj_layers)} features, got {len(features_list)}")
        projected = []
        for proj, feat in zip(self.proj_layers, features_list):
            if feat.dim() != 2:
                raise ValueError(f"Each feature must have shape [B, D], got {tuple(feat.shape)}")
            projected.append(proj(feat))
        values = torch.stack(projected, dim=1)
        fused, weights = self.selector(values, return_weights=True)
        fused = self.post(fused)
        if self.stable_residual:
            base_feature = self.base_norm(values.mean(dim=1))
            alpha = torch.sigmoid(self.residual_logit).to(fused.dtype)
            fused = (1.0 - alpha) * base_feature + alpha * fused
        if return_weights:
            return fused, weights
        return fused


class ResidualGateFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, gate_init: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate_linear.weight)
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.gate_linear.bias, math.log(gate_init / (1.0 - gate_init)))

    def forward(self, base_feature: torch.Tensor, residual_feature: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_linear(self.dropout(torch.cat([base_feature, residual_feature], dim=-1)))
        gate = torch.sigmoid(gate_logits)
        return base_feature + gate * (residual_feature - base_feature)

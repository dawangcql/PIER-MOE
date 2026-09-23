from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MLPExpert(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractionExpert(nn.Module):
    """Expert that models interactions among local granules."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int, n: int, d: int, num_heads: int = 4):
        super().__init__()
        if input_size != n * d:
            raise ValueError("input_size must equal n * d")
        if d % num_heads != 0:
            num_heads = 1
        self.n = n
        self.d = d
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = x.view(x.size(0), self.n, self.d)
        attn_output, _ = self.attn(local_features, local_features, local_features)
        features = self.norm(local_features + attn_output)
        return self.ffn(features.mean(dim=1))


class SelectionExpert(nn.Module):
    """Expert that selects the most useful local granules with learned weights."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int, n: int, d: int):
        super().__init__()
        if input_size != n * d:
            raise ValueError("input_size must equal n * d")
        self.n = n
        self.d = d
        self.scorer = nn.Sequential(nn.Linear(d, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.fc_out = nn.Linear(d, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = x.view(x.size(0), self.n, self.d)
        scores = self.scorer(local_features).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return self.fc_out((local_features * weights).sum(dim=1))


class GroupedMoE(nn.Module):
    """
    GLoMo-style grouped mixture of experts.

    Each group contains experts of the same type: interaction, selection, and MLP.
    Top-k routing happens inside each group, then a learned combiner mixes the group outputs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n: int,
        d: int,
        num_experts_per_type: int,
        k: int = 2,
    ):
        super().__init__()
        self.k = max(1, min(k, num_experts_per_type))
        self.num_experts_per_type = num_experts_per_type

        self.interaction_experts = nn.ModuleList(
            [InteractionExpert(input_size, output_size, hidden_size, n, d) for _ in range(num_experts_per_type)]
        )
        self.selection_experts = nn.ModuleList(
            [SelectionExpert(input_size, output_size, hidden_size, n, d) for _ in range(num_experts_per_type)]
        )
        self.mlp_experts = nn.ModuleList(
            [MLPExpert(input_size, output_size, hidden_size) for _ in range(num_experts_per_type)]
        )

        self.router_interaction = nn.Linear(input_size, num_experts_per_type)
        self.router_selection = nn.Linear(input_size, num_experts_per_type)
        self.router_mlp = nn.Linear(input_size, num_experts_per_type)
        self.group_combiner = nn.Linear(input_size, 3)

    def _get_group_output(self, x: torch.Tensor, router: nn.Linear, experts: nn.ModuleList) -> Tuple[torch.Tensor, torch.Tensor]:
        gating_logits = router(x)
        gating_probs = F.softmax(gating_logits, dim=1)
        top_k_probs, top_k_indices = torch.topk(gating_probs, self.k, dim=1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(x) for expert in experts], dim=1)
        output_dim = expert_outputs.size(-1)
        gather_index = top_k_indices.unsqueeze(-1).expand(-1, -1, output_dim)
        top_k_outputs = torch.gather(expert_outputs, 1, gather_index)
        weighted_output = (top_k_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)

        load_balance_loss = gating_probs.mean(dim=0).var(unbiased=False)
        return weighted_output, load_balance_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        interaction_output, interaction_loss = self._get_group_output(x, self.router_interaction, self.interaction_experts)
        selection_output, selection_loss = self._get_group_output(x, self.router_selection, self.selection_experts)
        mlp_output, mlp_loss = self._get_group_output(x, self.router_mlp, self.mlp_experts)

        group_weights = F.softmax(self.group_combiner(x), dim=1).unsqueeze(-1)
        group_outputs = torch.stack([interaction_output, selection_output, mlp_output], dim=1)
        final_output = (group_outputs * group_weights).sum(dim=1)
        moe_loss = interaction_loss + selection_loss + mlp_loss
        return final_output, moe_loss


def chunked_masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], granules: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B, T, D], got {tuple(x.shape)}")
    if mask is None:
        mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
    else:
        mask = mask.to(device=x.device, dtype=x.dtype)
    boundaries = torch.linspace(0, x.size(1), steps=granules + 1, device=x.device).round().long()
    pooled = []
    for idx in range(granules):
        start = int(boundaries[idx].item())
        end = int(boundaries[idx + 1].item())
        if end <= start:
            pooled.append(x.new_zeros(x.size(0), x.size(-1)))
            continue
        chunk = x[:, start:end, :]
        chunk_mask = mask[:, start:end].unsqueeze(-1)
        denom = chunk_mask.sum(dim=1).clamp_min(1.0)
        pooled.append((chunk * chunk_mask).sum(dim=1) / denom)
    return torch.stack(pooled, dim=1)


class LocalFeatureMoE(nn.Module):
    """Optional local branch that keeps GLoMo's fine-grained fusion idea."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        granules: int,
        num_experts_per_type: int,
        k: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.granules = granules
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.moe = GroupedMoE(
            input_size=hidden_dim * granules,
            output_size=hidden_dim,
            hidden_size=hidden_dim,
            n=granules,
            d=hidden_dim,
            num_experts_per_type=num_experts_per_type,
            k=k,
        )

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.proj(sequence)
        local_slots = chunked_masked_mean(sequence, mask, self.granules)
        return self.moe(torch.flatten(local_slots, start_dim=1))

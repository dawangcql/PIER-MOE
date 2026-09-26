
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _MLPExpert(nn.Module):
    """Lightweight MLP expert: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class EmotionGroupedMoE(nn.Module):
    """Polarity + Interaction + Shared Mixture of Experts for MSA / MER."""

    POS, NEG, NEU = 0, 1, 2
    UNQ, SYN, RED = 0, 1, 2

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_modalities: int = 3,
        dropout: float = 0.1,
        shared_alpha_init: float = 0.2,
        polarity_threshold: float = 0.1,
        mask_strategy: str = "random",
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if input_size % n_modalities != 0:
            raise ValueError(
                f"input_size {input_size} must be divisible by n_modalities {n_modalities}"
            )
        if mask_strategy not in {"random", "zero", "mean"}:
            raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

        self.input_size = input_size
        self.output_size = output_size
        self.n_modalities = n_modalities
        self.modal_chunk = input_size // n_modalities
        self.polarity_threshold = polarity_threshold
        self.mask_strategy = mask_strategy

        self.loss_weights = {
            "polarity_guide": 0.3,
            "interaction_pid": 0.1,
            "balance": 0.01,
        }
        if loss_weights:
            self.loss_weights.update(loss_weights)

        # Polarity group: {pos, neg, neu}
        self.polarity_experts = nn.ModuleList([
            _MLPExpert(input_size, output_size, hidden_size, dropout) for _ in range(3)
        ])
        # Interaction group: {uniqueness, synergy, redundancy}
        self.interaction_experts = nn.ModuleList([
            _MLPExpert(input_size, output_size, hidden_size, dropout) for _ in range(3)
        ])
        # Shared expert
        self.shared_expert = _MLPExpert(input_size, output_size, hidden_size, dropout)

        # Routers (3 experts each, full softmax — no top-k for clean Guide Task gradient)
        self.polarity_router = nn.Linear(input_size, 3)
        self.interaction_router = nn.Linear(input_size, 3)
        # Group combiner: how much polarity vs. interaction at each sample
        self.group_combiner = nn.Linear(input_size, 2)

        # Shared expert weight: sigmoid(alpha_logit). Initialised so sigmoid = shared_alpha_init.
        clamped = float(max(min(shared_alpha_init, 1.0 - 1e-4), 1e-4))
        self.shared_alpha_logit = nn.Parameter(
            torch.tensor(math.log(clamped / (1.0 - clamped)))
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expert_stack(self, experts: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """Run a list of experts and stack outputs as [B, E, D]."""
        return torch.stack([e(x) for e in experts], dim=1)

    @staticmethod
    def _mix(stack: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """gate [B, E], stack [B, E, D] -> [B, D]."""
        return (stack * gate.unsqueeze(-1)).sum(dim=1)

    def _make_mask_chunk(self, ref: torch.Tensor) -> torch.Tensor:
        if self.mask_strategy == "random":
            return torch.randn_like(ref)
        if self.mask_strategy == "zero":
            return torch.zeros_like(ref)
        # "mean": replace with batch mean, broadcast back
        mean = ref.mean(dim=0, keepdim=True)
        return mean.expand_as(ref)

    def _mask_one_modality(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Leave-one-out: replace modality m's slice with noise/zero/mean."""
        x_out = x.clone()
        s, e = m * self.modal_chunk, (m + 1) * self.modal_chunk
        x_out[:, s:e] = self._make_mask_chunk(x_out[:, s:e])
        return x_out

    def _mask_keep_only(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Leave-only-in: keep modality m, mask all others."""
        x_out = x.clone()
        for i in range(self.n_modalities):
            if i == m:
                continue
            s, e = i * self.modal_chunk, (i + 1) * self.modal_chunk
            x_out[:, s:e] = self._make_mask_chunk(x_out[:, s:e])
        return x_out

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def _polarity_guide_loss(self, gate: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """PAMoE-style Guide Task: force the polarity router to pick the
        expert matching the label's sign.

        Args:
            gate:   [B, 3] full softmax over polarity experts (pos, neg, neu).
            labels: [B] or [B, 1] regression labels in [-3, 3].
        """
        labels = labels.view(-1).to(gate.device)
        target = torch.full_like(labels, fill_value=self.NEU, dtype=torch.long)
        target = torch.where(
            labels > self.polarity_threshold,
            torch.full_like(target, self.POS),
            target,
        )
        target = torch.where(
            labels < -self.polarity_threshold,
            torch.full_like(target, self.NEG),
            target,
        )
        return F.nll_loss(torch.log(gate.clamp_min(1e-8)), target)

    def _interaction_pid_loss(
        self,
        x: torch.Tensor,
        full_int_stack: torch.Tensor,  # [B, 3, D]
    ) -> torch.Tensor:
        """I2MoE-inspired weakly-supervised PID loss with feature-level
        cosine similarity:

        - Uniqueness: full output should be SIMILAR to leave-only-in
          variants (single-modal info preserved).
        - Synergy:    full output should DIFFER from leave-one-out
          variants (joint info breaks under any mask).
        - Redundancy: full output should be SIMILAR to leave-one-out
          variants (shared info survives any single mask).
        """
        # ---- Uniqueness: leave-only-in (n_modalities forwards) ----
        loi_outs = []
        for m in range(self.n_modalities):
            x_loi = self._mask_keep_only(x, m)
            loi_outs.append(self.interaction_experts[self.UNQ](x_loi))
        loi_stack = torch.stack(loi_outs, dim=1)  # [B, n, D]
        f_unq = full_int_stack[:, self.UNQ:self.UNQ + 1, :]  # [B, 1, D]
        cos_unq = F.cosine_similarity(f_unq, loi_stack, dim=-1)  # [B, n]
        l_unq = (1.0 - cos_unq).mean()

        # ---- Synergy + Redundancy: leave-one-out (n_modalities forwards) ----
        syn_outs, red_outs = [], []
        for m in range(self.n_modalities):
            x_loo = self._mask_one_modality(x, m)
            syn_outs.append(self.interaction_experts[self.SYN](x_loo))
            red_outs.append(self.interaction_experts[self.RED](x_loo))
        syn_stack = torch.stack(syn_outs, dim=1)  # [B, n, D]
        red_stack = torch.stack(red_outs, dim=1)  # [B, n, D]

        f_syn = full_int_stack[:, self.SYN:self.SYN + 1, :]
        f_red = full_int_stack[:, self.RED:self.RED + 1, :]
        cos_syn = F.cosine_similarity(f_syn, syn_stack, dim=-1)  # [B, n]
        cos_red = F.cosine_similarity(f_red, red_stack, dim=-1)  # [B, n]

        # Synergy: minimize similarity (push outputs apart). We clamp at 0
        # to avoid pulling already-orthogonal pairs further apart.
        l_syn = cos_syn.clamp_min(0.0).mean()
        l_red = (1.0 - cos_red).mean()

        return l_unq + l_syn + l_red

    @staticmethod
    def _balance_loss(*gates: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer-style: minimize variance of mean usage per
        expert, summed over routers."""
        total = gates[0].new_tensor(0.0)
        for g in gates:
            total = total + g.mean(dim=0).var(unbiased=False)
        return total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_pid: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x:           [B, input_size] features (concat of n_modalities chunks).
            labels:      [B] or [B, 1] regression labels (only used in training
                         to drive the polarity Guide Task). If None, the
                         polarity guide loss is skipped.
            compute_pid: whether to run the extra forward passes for the
                         interaction PID loss. Disable to save compute on
                         non-training paths.

        Returns:
            output:       [B, output_size] fused MoE output.
            total_loss:   scalar tensor — already weighted sum of aux losses.
            losses:       dict of named raw aux losses (polarity_guide,
                         interaction_pid, balance) for logging.
        """
        # G1: Polarity ---------------------------------------------------
        pol_logits = self.polarity_router(x)
        pol_gate = F.softmax(pol_logits, dim=1)
        pol_stack = self._expert_stack(self.polarity_experts, x)
        pol_out = self._mix(pol_stack, pol_gate)

        # G2: Interaction ------------------------------------------------
        int_logits = self.interaction_router(x)
        int_gate = F.softmax(int_logits, dim=1)
        int_stack = self._expert_stack(self.interaction_experts, x)
        int_out = self._mix(int_stack, int_gate)

        # Shared expert (always on) -------------------------------------
        shared_out = self.shared_expert(x)

        # Group combination ---------------------------------------------
        group_w = F.softmax(self.group_combiner(x), dim=1)  # [B, 2]
        routed = group_w[:, 0:1] * pol_out + group_w[:, 1:2] * int_out
        alpha = torch.sigmoid(self.shared_alpha_logit)
        final = alpha * shared_out + (1.0 - alpha) * routed

        # Aux losses -----------------------------------------------------
        zero = x.new_tensor(0.0)
        losses: Dict[str, torch.Tensor] = {
            "polarity_guide": zero,
            "interaction_pid": zero,
            "balance": zero,
        }
        if self.training:
            if labels is not None:
                losses["polarity_guide"] = self._polarity_guide_loss(pol_gate, labels)
            if compute_pid:
                losses["interaction_pid"] = self._interaction_pid_loss(x, int_stack)
        losses["balance"] = self._balance_loss(pol_gate, int_gate)

        total = (
            self.loss_weights["polarity_guide"] * losses["polarity_guide"]
            + self.loss_weights["interaction_pid"] * losses["interaction_pid"]
            + self.loss_weights["balance"] * losses["balance"]
        )
        return final, total, losses


# ---------------------------------------------------------------------------
# Optional convenience: a polarity-only MoE for the local (per-modality) path
# ---------------------------------------------------------------------------

class PolarityOnlyMoE(nn.Module):
    """A lightweight polarity-only MoE for single-modality / local features.

    The Interaction group does not apply when only ONE modality is in the
    input (no cross-modal interactions are possible at that point), so this
    variant keeps just Polarity + Shared.
    """

    POS, NEG, NEU = 0, 1, 2

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        shared_alpha_init: float = 0.2,
        polarity_threshold: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.polarity_threshold = polarity_threshold
        self.loss_weights = {"polarity_guide": 0.3, "balance": 0.01}
        if loss_weights:
            self.loss_weights.update(loss_weights)

        self.polarity_experts = nn.ModuleList([
            _MLPExpert(input_size, output_size, hidden_size, dropout) for _ in range(3)
        ])
        self.shared_expert = _MLPExpert(input_size, output_size, hidden_size, dropout)
        self.polarity_router = nn.Linear(input_size, 3)

        clamped = float(max(min(shared_alpha_init, 1.0 - 1e-4), 1e-4))
        self.shared_alpha_logit = nn.Parameter(
            torch.tensor(math.log(clamped / (1.0 - clamped)))
        )

    def _polarity_guide_loss(self, gate: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).to(gate.device)
        target = torch.full_like(labels, fill_value=self.NEU, dtype=torch.long)
        target = torch.where(
            labels > self.polarity_threshold,
            torch.full_like(target, self.POS),
            target,
        )
        target = torch.where(
            labels < -self.polarity_threshold,
            torch.full_like(target, self.NEG),
            target,
        )
        return F.nll_loss(torch.log(gate.clamp_min(1e-8)), target)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        gate = F.softmax(self.polarity_router(x), dim=1)
        stack = torch.stack([e(x) for e in self.polarity_experts], dim=1)
        routed = (stack * gate.unsqueeze(-1)).sum(dim=1)
        shared_out = self.shared_expert(x)
        alpha = torch.sigmoid(self.shared_alpha_logit)
        final = alpha * shared_out + (1.0 - alpha) * routed

        zero = x.new_tensor(0.0)
        losses = {"polarity_guide": zero, "balance": zero}
        if self.training and labels is not None:
            losses["polarity_guide"] = self._polarity_guide_loss(gate, labels)
        losses["balance"] = gate.mean(dim=0).var(unbiased=False)

        total = (
            self.loss_weights["polarity_guide"] * losses["polarity_guide"]
            + self.loss_weights["balance"] * losses["balance"]
        )
        return final, total, losses


__all__ = ["EmotionGroupedMoE", "PolarityOnlyMoE"]

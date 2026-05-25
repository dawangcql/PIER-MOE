from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


PLACEHOLDER = "accept will be available"


class EmotionGroupedMoE(nn.Module):
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
        polarity_label_mode: str = "soft",
        polarity_strong_threshold: float = 2.0,
        mask_strategy: str = "random",
        loss_weights: Optional[Dict[str, float]] = None,
        desc_router_in_dim: int = 0,
        desc_logit_scale_init: float = 1.0,
        use_triplet_uniqueness: bool = True,
        triplet_margin: float = 1.0,
        interaction_router_target_mode: str = "none",
        interaction_router_temperature: float = 0.7,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        if input_size % n_modalities != 0:
            raise ValueError(
                f"input_size {input_size} must be divisible by n_modalities {n_modalities}"
            )
        if mask_strategy not in {"random", "zero", "mean"}:
            raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
        if polarity_label_mode not in {"hard", "soft"}:
            raise ValueError(f"Unknown polarity_label_mode: {polarity_label_mode}")
        if polarity_strong_threshold <= 0:
            raise ValueError("polarity_strong_threshold must be positive.")
        if polarity_strong_threshold <= polarity_threshold:
            raise ValueError("polarity_strong_threshold must be greater than polarity_threshold.")
        if interaction_router_target_mode not in {"none", "expert_cosine"}:
            raise ValueError(
                "interaction_router_target_mode must be 'none' or 'expert_cosine'."
            )
        if interaction_router_temperature <= 0:
            raise ValueError("interaction_router_temperature must be positive.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")

        self.input_size = input_size
        self.output_size = output_size
        self.n_modalities = n_modalities
        self.modal_chunk = input_size // n_modalities
        self.polarity_threshold = polarity_threshold
        self.polarity_label_mode = polarity_label_mode
        self.polarity_strong_threshold = polarity_strong_threshold
        self.mask_strategy = mask_strategy
        self.desc_router_in_dim = max(0, int(desc_router_in_dim))
        self.use_triplet_uniqueness = bool(use_triplet_uniqueness)
        self.interaction_router_target_mode = interaction_router_target_mode
        self.interaction_router_temperature = interaction_router_temperature
        self.gate_temperature = gate_temperature

        self.loss_weights = {
            "polarity_guide": 0.3,
            "interaction_pid": 0.1,
            "interaction_router": 0.05,
            "balance": 0.01,
            "router_entropy": 0.01,
        }
        if loss_weights:
            self.loss_weights.update(loss_weights)

        # ----- experts: ALL consume x_pure (no description leakage) -----
        self.polarity_experts = nn.ModuleList(
            [_MLPExpert(input_size, output_size, hidden_size, dropout) for _ in range(3)]
        )
        self.interaction_experts = nn.ModuleList(
            [_MLPExpert(input_size, output_size, hidden_size, dropout) for _ in range(3)]
        )
        self.shared_expert = _MLPExpert(input_size, output_size, hidden_size, dropout)

        # ----- routers --------------------------------------------------
        # Polarity router never sees descriptions.
        self.polarity_router = nn.Linear(input_size, 3)
        # Interaction routing is decomposed into a pure-feature logits path
        # and an optional description logits path. This keeps descriptions
        # from being drowned out by the much wider x_pure vector.
        self.interaction_router = nn.Linear(input_size, 3)
        self.interaction_desc_router = (
            nn.Linear(self.desc_router_in_dim, 3) if self.desc_router_in_dim > 0 else None
        )
        if self.interaction_desc_router is not None:
            nn.init.zeros_(self.interaction_desc_router.bias)
        raw_desc_scale = math.log(math.expm1(max(float(desc_logit_scale_init), 1e-6)))
        self.desc_logit_scale = nn.Parameter(torch.tensor(raw_desc_scale))
        # Group combiner: pure-only, so the pol-vs-int mix is not biased by
        # whether a description happens to be present for the sample.
        self.group_combiner = nn.Linear(input_size, 2)

        clamped = float(max(min(shared_alpha_init, 1.0 - 1e-4), 1e-4))
        self.shared_alpha_logit = nn.Parameter(
            torch.tensor(math.log(clamped / (1.0 - clamped)))
        )

        if self.use_triplet_uniqueness:
            self.triplet_loss_fn = nn.TripletMarginLoss(margin=triplet_margin, p=2)
        else:
            self.triplet_loss_fn = None


    def _polarity_guide_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).to(logits.device)
        scaled_logits = self._scaled_logits(logits)
        if self.polarity_label_mode == "soft":
            target = logits.new_zeros(labels.size(0), 3)
            threshold = float(self.polarity_threshold)
            strong = float(self.polarity_strong_threshold)
            strength = ((labels.abs() - threshold) / (strong - threshold)).clamp(0.0, 1.0)
            pos_mask = labels > threshold
            neg_mask = labels < -threshold
            target[:, self.NEU] = 1.0 - strength
            target[:, self.POS] = torch.where(pos_mask, strength, torch.zeros_like(strength))
            target[:, self.NEG] = torch.where(neg_mask, strength, torch.zeros_like(strength))
            # Tiny numerical guard for labels that fall inside the neutral core.
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return -(target * F.log_softmax(scaled_logits, dim=1)).sum(dim=1).mean()

        # ==================================
        # accept will be available
        # ==================================

        return F.cross_entropy(scaled_logits, target)

    def _interaction_pid_loss(
        self,
        x: torch.Tensor,
        full_int_stack: torch.Tensor,
        router_target_mode: str = "none",
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        loi_outs = []
        for m in range(self.n_modalities):
            loi_outs.append(self.interaction_experts[self.UNQ](self._mask_keep_only(x, m)))
        loi_stack = torch.stack(loi_outs, dim=1)
        f_unq = full_int_stack[:, self.UNQ:self.UNQ + 1, :]
        f_syn = full_int_stack[:, self.SYN:self.SYN + 1, :]
        f_red = full_int_stack[:, self.RED:self.RED + 1, :]

        # ---- Uniqueness ------------------------------------------------
        if self.use_triplet_uniqueness:
            # Anchor: full UNQ ; Positive: each LOI(m) ; Negative: full SYN.
            anchor = f_unq.squeeze(1)               # [B, D]
            negative = f_syn.squeeze(1)             # [B, D]
            l_unq = anchor.new_tensor(0.0)
            for m in range(self.n_modalities):
                l_unq = l_unq + self.triplet_loss_fn(anchor, loi_outs[m], negative)
            l_unq = l_unq / float(self.n_modalities)
        else:
            cos_unq = F.cosine_similarity(f_unq, loi_stack, dim=-1)
            l_unq = (1.0 - cos_unq).mean()

        # ---- Synergy & Redundancy (cosine, unchanged from v2) ----------
        syn_outs, red_outs = [], []
        for m in range(self.n_modalities):
            x_loo = self._mask_one_modality(x, m)
            syn_outs.append(self.interaction_experts[self.SYN](x_loo))
            red_outs.append(self.interaction_experts[self.RED](x_loo))
        syn_stack = torch.stack(syn_outs, dim=1)
        red_stack = torch.stack(red_outs, dim=1)

        # ==================================
        # accept will be available
        # ==================================

        return loss, target.detach()
    def forward(
        self,
        x: torch.Tensor,
        desc_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_pid: bool = True,
        aux_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError(PLACEHOLDER)


__all__ = ["EmotionGroupedMoE"]

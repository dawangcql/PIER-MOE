
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _MLPExpert(nn.Module):
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expert_stack(self, experts: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([e(x) for e in experts], dim=1)

    @staticmethod
    def _mix(stack: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return (stack * gate.unsqueeze(-1)).sum(dim=1)

    def _scaled_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.gate_temperature

    def _router_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self._scaled_logits(logits), dim=1)

    def _interaction_logits(
        self,
        x: torch.Tensor,
        desc_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_logits = self.interaction_router(x)
        if self.interaction_desc_router is None:
            desc_logits = x_logits.new_zeros(x_logits.shape)
            return x_logits, x_logits, desc_logits
        if desc_features is None:
            desc_features = x.new_zeros(x.size(0), self.desc_router_in_dim)
        desc_logits = self.interaction_desc_router(desc_features)
        scale = F.softplus(self.desc_logit_scale).to(dtype=x_logits.dtype)
        return x_logits + scale * desc_logits, x_logits, desc_logits

    def _make_mask_chunk(self, ref: torch.Tensor) -> torch.Tensor:
        if self.mask_strategy == "random":
            return torch.randn_like(ref)
        if self.mask_strategy == "zero":
            return torch.zeros_like(ref)
        mean = ref.mean(dim=0, keepdim=True)
        return mean.expand_as(ref)

    def _mask_one_modality(self, x: torch.Tensor, m: int) -> torch.Tensor:
        x_out = x.clone()
        s, e = m * self.modal_chunk, (m + 1) * self.modal_chunk
        x_out[:, s:e] = self._make_mask_chunk(x_out[:, s:e])
        return x_out

    def _mask_keep_only(self, x: torch.Tensor, m: int) -> torch.Tensor:
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

        # ---- Synergy & Redundancy (cosine) ------------------------------
        syn_outs, red_outs = [], []
        for m in range(self.n_modalities):
            x_loo = self._mask_one_modality(x, m)
            syn_outs.append(self.interaction_experts[self.SYN](x_loo))
            red_outs.append(self.interaction_experts[self.RED](x_loo))
        syn_stack = torch.stack(syn_outs, dim=1)
        red_stack = torch.stack(red_outs, dim=1)
        cos_syn = F.cosine_similarity(f_syn, syn_stack, dim=-1)
        cos_red = F.cosine_similarity(f_red, red_stack, dim=-1)
        l_syn = cos_syn.clamp_min(0.0).mean()
        l_red = (1.0 - cos_red).mean()
        loss = l_unq + l_syn + l_red
        if router_target_mode == "none":
            return loss
        if router_target_mode != "expert_cosine":
            raise ValueError(f"Unknown interaction router target mode: {router_target_mode}")

        with torch.no_grad():
            unq_score = F.cosine_similarity(f_unq, loi_stack, dim=-1).mean(dim=1)
            syn_score = (1.0 - cos_syn).mean(dim=1)
            red_score = cos_red.mean(dim=1)
            scores = torch.stack([unq_score, syn_score, red_score], dim=1)
            target = F.softmax(scores / self.interaction_router_temperature, dim=1)
        return loss, target.detach()

    @staticmethod
    def _balance_loss(*gates: torch.Tensor) -> torch.Tensor:
        total = gates[0].new_tensor(0.0)
        for g in gates:
            n = g.size(1)
            uniform = g.new_full((n,), 1.0 / float(n))
            total = total + (g.mean(dim=0) - uniform).pow(2).mean()
        return total

    @staticmethod
    def _router_entropy_loss(*gates: torch.Tensor) -> torch.Tensor:
        total = gates[0].new_tensor(0.0)
        for g in gates:
            entropy = -(g.clamp_min(1e-8) * g.clamp_min(1e-8).log()).sum(dim=1).mean()
            total = total + (math.log(g.size(1)) - entropy)
        return total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        desc_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_pid: bool = True,
        aux_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x:              [B, input_size] — pure (text || audio || visual) concat.
            desc_features:  [B, desc_router_in_dim] — projected description
                            embedding fed ONLY to the Interaction Router.
                            If None, behaves like v2 (no desc).
            labels:         regression labels for the Polarity Guide Task.
            aux_scale:      MoE-aux warmup factor from the trainer.
        """
        # ---- Polarity (pure) ------------------------------------------
        pol_logits = self.polarity_router(x)
        pol_gate = self._router_softmax(pol_logits)
        pol_stack = self._expert_stack(self.polarity_experts, x)
        pol_out = self._mix(pol_stack, pol_gate)

        # ---- Interaction (description has an explicit logits branch) ---
        int_logits, int_logits_x, int_logits_desc = self._interaction_logits(x, desc_features)
        int_gate = self._router_softmax(int_logits)
        int_stack = self._expert_stack(self.interaction_experts, x)  # experts: x_pure
        int_out = self._mix(int_stack, int_gate)

        # ---- Shared (pure) --------------------------------------------
        shared_out = self.shared_expert(x)

        group_w = F.softmax(self.group_combiner(x), dim=1)
        routed = group_w[:, 0:1] * pol_out + group_w[:, 1:2] * int_out
        alpha = torch.sigmoid(self.shared_alpha_logit)
        final = alpha * shared_out + (1.0 - alpha) * routed

        zero = x.new_tensor(0.0)
        losses: Dict[str, torch.Tensor] = {
            "polarity_guide": zero,
            "interaction_pid": zero,
            "interaction_router": zero,
            "balance": zero,
            "router_entropy": zero,
        }
        if self.training:
            if labels is not None:
                losses["polarity_guide"] = self._polarity_guide_loss(pol_logits, labels)
            if compute_pid:
                want_router_target = (
                    self.loss_weights.get("interaction_router", 0.0) > 0
                    and self.interaction_router_target_mode != "none"
                )
                pid = self._interaction_pid_loss(
                    x,
                    int_stack,
                    router_target_mode=(
                        self.interaction_router_target_mode if want_router_target else "none"
                    ),
                )
                if want_router_target:
                    losses["interaction_pid"], int_target = pid
                    losses["interaction_router"] = F.kl_div(
                        F.log_softmax(self._scaled_logits(int_logits), dim=1),
                        int_target,
                        reduction="batchmean",
                    )
                else:
                    losses["interaction_pid"] = pid
        losses["balance"] = self._balance_loss(pol_gate, int_gate)
        losses["router_entropy"] = self._router_entropy_loss(pol_gate, int_gate)

        total = (
            aux_scale * self.loss_weights["polarity_guide"] * losses["polarity_guide"]
            + aux_scale * self.loss_weights["interaction_pid"] * losses["interaction_pid"]
            + aux_scale * self.loss_weights["interaction_router"] * losses["interaction_router"]
            + self.loss_weights["balance"] * losses["balance"]
            + self.loss_weights["router_entropy"] * losses["router_entropy"]
        )
        if not return_diagnostics:
            return final, total, losses

        if self.interaction_desc_router is not None:
            int_logits_no_desc = int_logits_x
            int_gate_no_desc = self._router_softmax(int_logits_no_desc)
            desc_norm = (
                desc_features.norm(dim=-1)
                if desc_features is not None
                else x.new_zeros(x.size(0))
            )
        else:
            int_logits_no_desc = int_logits
            int_gate_no_desc = int_gate
            desc_norm = x.new_zeros(x.size(0))
        desc_scale = F.softplus(self.desc_logit_scale).to(dtype=int_logits_x.dtype)
        int_logits_desc_scaled = desc_scale * int_logits_desc
        int_logits_x_norm = int_logits_x.norm(dim=-1)
        int_logits_desc_scaled_norm = int_logits_desc_scaled.norm(dim=-1)
        int_top2 = int_gate.topk(k=2, dim=1).values
        diagnostics = {
            "pol_logits": pol_logits,
            "pol_gate": pol_gate,
            "pol_gate_mean": pol_gate.mean(dim=0),
            "selected_pol_expert": pol_gate.argmax(dim=1),
            "pol_selected_hist": torch.bincount(
                pol_gate.argmax(dim=1), minlength=3
            ).float() / x.size(0),
            "int_logits": int_logits,
            "int_logits_x": int_logits_x,
            "int_logits_desc": int_logits_desc,
            "int_logits_desc_scaled": int_logits_desc_scaled,
            "int_logits_x_norm": int_logits_x_norm,
            "int_logits_desc_scaled_norm": int_logits_desc_scaled_norm,
            "desc_to_x_logit_norm_ratio": int_logits_desc_scaled_norm
            / int_logits_x_norm.clamp_min(1e-8),
            "int_gate": int_gate,
            "int_gate_mean": int_gate.mean(dim=0),
            "int_logits_no_desc": int_logits_no_desc,
            "int_gate_no_desc": int_gate_no_desc,
            "desc_gate_delta_l1": (int_gate - int_gate_no_desc).abs().sum(dim=1),
            "selected_int_expert": int_gate.argmax(dim=1),
            "int_selected_hist": torch.bincount(
                int_gate.argmax(dim=1), minlength=3
            ).float() / x.size(0),
            "int_gate_margin_top1_top2": int_top2[:, 0] - int_top2[:, 1],
            "group_w": group_w,
            "shared_alpha": alpha.expand(x.size(0)),
            "desc_logit_scale": F.softplus(self.desc_logit_scale).expand(x.size(0)),
            "desc_proj_norm": desc_norm,
            "x_pure_norm": x.norm(dim=-1),
        }
        return final, total, losses, diagnostics


__all__ = ["EmotionGroupedMoE"]

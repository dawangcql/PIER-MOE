from __future__ import annotations

from dataclasses import dataclass

from pier_moe_core.config import PIERMoEConfig as BasePIERMoEConfig


@dataclass
class PIERMoEConfig(BasePIERMoEConfig):
    # ---- description routing strategy --------------------------------
    route_desc_to_interaction_only: bool = True
    desc_router_proj_dim: int = 64
    desc_logit_scale_init: float = 1.0

    # ---- I2MoE-style triplet uniqueness loss --------------------------
    use_triplet_uniqueness: bool = True
    triplet_margin: float = 1.0

    # ---- typed-router supervision -------------------------------------
    moe_polarity_label_mode: str = "soft"  # 'hard' | 'soft'
    moe_polarity_threshold: float = 0.2
    moe_polarity_strong_threshold: float = 1.0
    moe_interaction_router_weight: float = 0.0
    moe_interaction_router_target_mode: str = "none"  # 'none' | 'expert_cosine'
    moe_interaction_router_temperature: float = 0.7
    moe_router_entropy_weight: float = 0.01
    moe_gate_temperature: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.desc_router_proj_dim <= 0:
            raise ValueError("desc_router_proj_dim must be positive.")
        if self.desc_logit_scale_init < 0:
            raise ValueError("desc_logit_scale_init must be non-negative.")
        if self.triplet_margin <= 0:
            raise ValueError("triplet_margin must be positive.")
        if self.moe_polarity_label_mode not in {"hard", "soft"}:
            raise ValueError("moe_polarity_label_mode must be 'hard' or 'soft'.")
        if self.moe_polarity_strong_threshold <= 0:
            raise ValueError("moe_polarity_strong_threshold must be positive.")
        if self.moe_interaction_router_weight < 0:
            raise ValueError("moe_interaction_router_weight must be non-negative.")
        if self.moe_interaction_router_target_mode not in {"none", "expert_cosine"}:
            raise ValueError(
                "moe_interaction_router_target_mode must be 'none' or 'expert_cosine'."
            )
        if self.moe_interaction_router_temperature <= 0:
            raise ValueError("moe_interaction_router_temperature must be positive.")
        if self.moe_router_entropy_weight < 0:
            raise ValueError("moe_router_entropy_weight must be non-negative.")
        if self.moe_gate_temperature <= 0:
            raise ValueError("moe_gate_temperature must be positive.")

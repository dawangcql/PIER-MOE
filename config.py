from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIERMoEConfig:
    text_model_name: str = "models/Baichuan-13B"
    audio_model_name_or_path: str = "models/hubert-large"
    desc_text_model_name: str = "models/Baichuan-13B"
    hf_local_only: bool = True
    trust_remote_code: bool = True
    backbone_torch_dtype: str = "auto"

    hidden_dim: int = 512
    visual_dim: int = 768
    desc_dim: int = 256
    dropout: float = 0.2
    return_auxiliary_heads: bool = True

    attnres_num_blocks: int = 4
    attnres_zero_init: bool = True
    attnres_init_alpha: float = 0.05

    desc_router_proj_dim: int = 64
    desc_logit_scale_init: float = 1.0

    moe_polarity_guide_weight: float = 0.3
    moe_interaction_pid_weight: float = 0.1
    moe_balance_weight: float = 0.01
    moe_router_entropy_weight: float = 0.01
    moe_shared_alpha_init: float = 0.2
    moe_polarity_threshold: float = 0.2
    moe_polarity_strong_threshold: float = 1.0
    use_triplet_uniqueness: bool = True
    triplet_margin: float = 1.0

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.visual_dim <= 0:
            raise ValueError("visual_dim must be positive.")
        if self.desc_dim <= 0:
            raise ValueError("desc_dim must be positive.")
        if self.desc_router_proj_dim <= 0:
            raise ValueError("desc_router_proj_dim must be positive.")
        if self.desc_logit_scale_init < 0:
            raise ValueError("desc_logit_scale_init must be non-negative.")
        if self.attnres_num_blocks <= 0:
            raise ValueError("attnres_num_blocks must be positive.")
        if self.moe_polarity_strong_threshold <= self.moe_polarity_threshold:
            raise ValueError("moe_polarity_strong_threshold must be greater than moe_polarity_threshold.")
        if self.triplet_margin <= 0:
            raise ValueError("triplet_margin must be positive.")
        if self.backbone_torch_dtype not in {"auto", "float16", "bfloat16", "float32", "none"}:
            raise ValueError("backbone_torch_dtype must be one of: auto, float16, bfloat16, float32, none.")

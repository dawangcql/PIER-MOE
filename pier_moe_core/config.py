from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIERMoEConfig:
    """Configuration for the simplified PIERMoE multimodal emotion model."""

    text_model_name: str = "/root/autodl-tmp/models/roberta-large"
    audio_model_name_or_path: str = "/root/autodl-tmp/models/wavlm-large"
    desc_text_model_name: str = "/root/autodl-tmp/models/roberta-large"
    hf_local_only: bool = True

    hidden_dim: int = 512
    visual_dim: int = 768
    openface_dim: int = 63
    desc_dim: int = 256
    dropout: float = 0.3

    use_context: bool = True
    use_visual: bool = True
    use_desc: bool = True
    freeze_desc_encoder: bool = True

    use_attnres: bool = True
    attnres_type: str = "block"
    attnres_num_blocks: int = 4
    attnres_zero_init: bool = True
    attnres_init_alpha: float = 0.05
    attnres_gate_init: float = 0.05

    use_grouped_moe: bool = True
    use_local_moe: bool = False
    moe_num_experts_per_type: int = 3
    moe_k: int = 2
    moe_balance_weight: float = 0.01
    local_gran_text: int = 3
    local_gran_audio: int = 3
    local_gran_visual: int = 3

    # ---- EmotionGroupedMoE (Polarity + Interaction + Shared) -----------
    # When True, the legacy GroupedMoE is replaced by EmotionGroupedMoE.
    use_emotion_moe: bool = True
    moe_polarity_guide_weight: float = 0.3
    moe_interaction_pid_weight: float = 0.1
    moe_shared_alpha_init: float = 0.2
    moe_polarity_threshold: float = 0.1
    moe_mask_strategy: str = "random"   # 'random' | 'zero' | 'mean'

    return_auxiliary_heads: bool = True

    def validate(self) -> None:
        if self.attnres_type not in {"full", "block"}:
            raise ValueError("attnres_type must be 'full' or 'block'.")
        if self.moe_mask_strategy not in {"random", "zero", "mean"}:
            raise ValueError("moe_mask_strategy must be 'random', 'zero', or 'mean'.")

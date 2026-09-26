
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from pier_moe_core.attention_residuals import ProjectedAttentionResidualFusion
from pier_moe_core.model import PIERMoEForMER as BasePIERMoEForMER

from .config import PIERMoEConfig
from .description_fusion import IndependentDescriptionEncoder
from .grouped_moe import EmotionGroupedMoE


class PIERMoEForMER(BasePIERMoEForMER):
    def __init__(self, config: PIERMoEConfig):
        super().__init__(config)
        if not isinstance(config, PIERMoEConfig):
            raise TypeError("PIERMoEForMER requires PIERMoEConfig.")
        self.cfg = config

        if self.use_desc:
            self.desc_fusion = IndependentDescriptionEncoder(
                desc_model_name=config.desc_text_model_name,
                audio_raw_dim=self.hidden_dim,
                visual_raw_dim=self.hidden_dim,
                desc_dim=config.desc_dim,
                freeze_text_encoder=config.freeze_desc_encoder,
                dropout=config.dropout,
                hf_local_only=config.hf_local_only,
            )
            # Project (desc_audio || desc_visual) down before feeding it to
            # the Interaction Router. Two desc tokens × desc_dim is too wide
            # for a 3-class router; this keeps capacity reasonable.
            if config.route_desc_to_interaction_only:
                self.desc_router_proj = nn.Sequential(
                    nn.LayerNorm(config.desc_dim * 2),
                    nn.Linear(config.desc_dim * 2, config.desc_router_proj_dim, bias=False),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
                self.desc_audio_feature_proj = None
                self.desc_visual_feature_proj = None
            else:
                self.desc_router_proj = None
                self.desc_audio_feature_proj = nn.Sequential(
                    nn.LayerNorm(config.desc_dim),
                    nn.Linear(config.desc_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
                self.desc_visual_feature_proj = nn.Sequential(
                    nn.LayerNorm(config.desc_dim),
                    nn.Linear(config.desc_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
            desc_router_in_dim = (
                config.desc_router_proj_dim if config.route_desc_to_interaction_only else 0
            )
        else:
            self.desc_fusion = None
            self.desc_router_proj = None
            self.desc_audio_feature_proj = None
            self.desc_visual_feature_proj = None
            desc_router_in_dim = 0

        if not self.use_desc:
            self.audio_desc_attnres = None
            self.visual_desc_attnres = None
        elif config.route_desc_to_interaction_only:
            self.audio_desc_attnres = None
            self.visual_desc_attnres = None
        else:
            attn_kwargs = dict(
                dropout=config.dropout,
                mode=config.attnres_type,
                num_blocks=config.attnres_num_blocks,
                zero_init=config.attnres_zero_init,
                stable_residual=True,
                residual_init_alpha=config.attnres_init_alpha,
            )
            self.audio_desc_attnres = ProjectedAttentionResidualFusion(
                [self.hidden_dim, self.hidden_dim], self.hidden_dim, **attn_kwargs
            )
            self.visual_desc_attnres = ProjectedAttentionResidualFusion(
                [self.hidden_dim, self.hidden_dim], self.hidden_dim, **attn_kwargs
            )

        if self.use_grouped_moe and config.use_emotion_moe:
            self.global_moe = EmotionGroupedMoE(
                input_size=self.hidden_dim * 3,
                output_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                n_modalities=3,
                dropout=config.dropout,
                shared_alpha_init=config.moe_shared_alpha_init,
                polarity_threshold=config.moe_polarity_threshold,
                polarity_label_mode=config.moe_polarity_label_mode,
                polarity_strong_threshold=config.moe_polarity_strong_threshold,
                mask_strategy=config.moe_mask_strategy,
                loss_weights={
                    "polarity_guide": config.moe_polarity_guide_weight,
                    "interaction_pid": config.moe_interaction_pid_weight,
                    "interaction_router": config.moe_interaction_router_weight,
                    "balance": config.moe_balance_weight,
                    "router_entropy": config.moe_router_entropy_weight,
                },
                desc_router_in_dim=desc_router_in_dim,
                desc_logit_scale_init=config.desc_logit_scale_init,
                use_triplet_uniqueness=config.use_triplet_uniqueness,
                triplet_margin=config.triplet_margin,
                interaction_router_target_mode=config.moe_interaction_router_target_mode,
                interaction_router_temperature=config.moe_interaction_router_temperature,
                gate_temperature=config.moe_gate_temperature,
            )

    def _run_global_moe(
        self,
        global_cat: torch.Tensor,
        desc_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        moe_aux_scale: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        if isinstance(self.global_moe, EmotionGroupedMoE):
            result = self.global_moe(
                global_cat,
                desc_features=desc_features,
                labels=labels,
                aux_scale=moe_aux_scale,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                out, scalar_loss, raw, diagnostics = result
                return out, scalar_loss, raw, diagnostics
            out, scalar_loss, raw = result
            return out, scalar_loss, raw, {}
        out, scalar_loss = self.global_moe(global_cat)
        return out, scalar_loss, {}, {}

    def _build_moe_feature(
        self,
        text_global,
        audio_global,
        visual_global,
        desc_features,
        labels: Optional[torch.Tensor] = None,
        moe_aux_scale: float = 1.0,
        return_diagnostics: bool = False,
    ):
        zero_loss = text_global.new_tensor(0.0)
        if not self.use_grouped_moe:
            return None, zero_loss, {}, {}
        global_cat = torch.cat([text_global, audio_global, visual_global], dim=-1)
        return self._run_global_moe(
            global_cat,
            desc_features=desc_features,
            labels=labels,
            moe_aux_scale=moe_aux_scale,
            return_diagnostics=return_diagnostics,
        )

    def forward(
        self,
        text_inputs,
        text_mask,
        audio_inputs,
        audio_mask,
        visual_inputs=None,
        visual_mask=None,
        text_context_inputs=None,
        text_context_mask=None,
        audio_context_inputs=None,
        audio_context_mask=None,
        visual_context_inputs=None,
        visual_context_mask=None,
        audio_desc_tokens=None,
        audio_desc_masks=None,
        audio_desc_valid=None,
        visual_desc_tokens=None,
        visual_desc_masks=None,
        visual_desc_valid=None,
        openface_inputs=None,
        openface_valid=None,
        labels=None,
        moe_aux_scale: float = 1.0,
        moe_ablation_mode: str = "none",
        return_diagnostics: bool = False,
    ) -> dict:
        text_seq, text_global, text_seq_mask = self._encode_text(text_inputs, text_mask)
        audio_seq, audio_global, audio_seq_mask = self._encode_audio(audio_inputs, audio_mask)

        if self.use_visual:
            if visual_inputs is None:
                raise ValueError("visual_inputs is required when use_visual=True.")
            visual_seq, visual_global, visual_seq_mask = self.visual_encoder(visual_inputs, visual_mask)
        else:
            visual_seq = audio_seq.new_zeros(audio_seq.size(0), 1, self.hidden_dim)
            visual_global = audio_global.new_zeros(audio_global.shape)
            visual_seq_mask = torch.ones(audio_seq.size(0), 1, device=audio_seq.device, dtype=torch.bool)

        ctx = self._encode_optional_context(
            text_context_inputs, text_context_mask,
            audio_context_inputs, audio_context_mask,
            visual_context_inputs, visual_context_mask,
        )
        text_ctx, text_ctx_valid, audio_ctx, audio_ctx_valid, visual_ctx, visual_ctx_valid = ctx
        text_global = self._contextualize(text_global, text_ctx, text_ctx_valid, self.text_context_attnres)
        audio_global = self._contextualize(audio_global, audio_ctx, audio_ctx_valid, self.audio_context_attnres)
        visual_global = self._contextualize(visual_global, visual_ctx, visual_ctx_valid, self.visual_context_attnres)

        # Description is encoded independently. The default v3 path sends it
        # only to the interaction router; the v2-style ablation fuses it back
        # into audio/visual globals before MoE.
        zero = audio_global.new_tensor(0.0)
        if self.use_desc:
            desc_audio, desc_visual, aux_losses = self.desc_fusion(
                audio_global, visual_global,
                audio_desc_tokens, audio_desc_masks, audio_desc_valid,
                visual_desc_tokens, visual_desc_masks, visual_desc_valid,
            )
            if self.cfg.route_desc_to_interaction_only:
                desc_concat = torch.cat([desc_audio, desc_visual], dim=-1)
                desc_proj = self.desc_router_proj(desc_concat)
            else:
                desc_proj = None
                audio_desc_hidden = self.desc_audio_feature_proj(desc_audio)
                visual_desc_hidden = self.desc_visual_feature_proj(desc_visual)
                audio_global = self.audio_desc_attnres([audio_global, audio_desc_hidden])
                visual_global = self.visual_desc_attnres([visual_global, visual_desc_hidden])
        else:
            desc_proj = None
            aux_losses = {
                "audio_align": zero,
                "visual_align": zero,
                "visual_openface_align": zero,
            }

        moe_feature, moe_loss, moe_raw_losses, moe_diagnostics = self._build_moe_feature(
            text_global, audio_global, visual_global,
            desc_features=desc_proj,
            labels=labels,
            moe_aux_scale=moe_aux_scale,
            return_diagnostics=return_diagnostics,
        )

        if moe_feature is not None:
            if moe_ablation_mode == "none":
                pass
            elif moe_ablation_mode == "zero":
                moe_feature = torch.zeros_like(moe_feature)
            elif moe_ablation_mode == "mean":
                moe_feature = moe_feature.mean(dim=0, keepdim=True).expand_as(moe_feature)
            else:
                raise ValueError(f"Unknown moe_ablation_mode: {moe_ablation_mode}")

        attnres_sources = [text_global, audio_global, visual_global]
        if moe_feature is not None:
            attnres_sources.append(moe_feature)
        if return_diagnostics:
            multimodal_feature, final_fusion_weights = self.multimodal_attnres(
                attnres_sources,
                return_weights=True,
            )
        else:
            final_fusion_weights = None
            multimodal_feature = self.multimodal_attnres(attnres_sources)

        regression_pred = self.regression_head(multimodal_feature)
        outputs = {
            "M": regression_pred,
            "label_feature": multimodal_feature,
            "_aux_losses": aux_losses,
            "_moe_loss": moe_loss,
            "_moe_raw_losses": moe_raw_losses,
        }
        if self.config.return_auxiliary_heads:
            outputs["T"] = self.T_output_layers(text_global)
            outputs["A"] = self.A_output_layers(audio_global)
            outputs["V"] = self.V_output_layers(visual_global)
        if return_diagnostics:
            diagnostics = dict(moe_diagnostics)
            diagnostics["final_fusion_weights"] = final_fusion_weights
            diagnostics["final_source_names"] = (
                ["text", "audio", "visual", "moe"]
                if moe_feature is not None
                else ["text", "audio", "visual"]
            )
            diagnostics["desc_to_interaction_only"] = torch.tensor(
                float(self.cfg.route_desc_to_interaction_only),
                device=regression_pred.device,
            )
            outputs["_diagnostics"] = diagnostics
        return outputs


__all__ = ["PIERMoEForMER"]

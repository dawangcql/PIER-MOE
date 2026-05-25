from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

try:
    from .config import PIERMoEConfig
except ImportError:
    from config import PIERMoEConfig


def masked_mean_pool(features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return features.mean(dim=1)
    mask = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def sequence_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange < lengths.unsqueeze(1)


class VisualFeatureEncoder(nn.Module):
    def __init__(self, visual_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual_inputs: torch.Tensor,
        visual_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_inputs.dim() == 2:
            visual_inputs = visual_inputs.unsqueeze(1)
        if visual_mask is None:
            visual_mask = torch.ones(
                visual_inputs.size(0),
                visual_inputs.size(1),
                device=visual_inputs.device,
                dtype=torch.bool,
            )
        visual_seq = self.frame_proj(visual_inputs.float())
        visual_global = masked_mean_pool(visual_seq, visual_mask)
        return visual_seq, visual_global, visual_mask.bool()


class PIERMoEForMER(nn.Module):
    def __init__(self, config: PIERMoEConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.hidden_dim = config.hidden_dim

        model_kwargs = {
            "local_files_only": config.hf_local_only,
            "trust_remote_code": config.trust_remote_code,
        }
        torch_dtype = _resolve_torch_dtype(config.backbone_torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.text_model = AutoModel.from_pretrained(config.text_model_name, **model_kwargs)
        self.audio_model = AutoModel.from_pretrained(
            config.audio_model_name_or_path,
            **model_kwargs,
        )
        self._freeze_module(self.text_model)
        self._freeze_module(self.audio_model)

        text_hidden = _model_hidden_size(self.text_model)
        audio_hidden = _model_hidden_size(self.audio_model)
        self.text_pool_proj = nn.Sequential(
            nn.LayerNorm(text_hidden),
            nn.Linear(text_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_hidden),
            nn.Linear(audio_hidden, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.visual_encoder = VisualFeatureEncoder(
            config.visual_dim,
            self.hidden_dim,
            config.dropout,
        )

        share_desc_backbone = config.desc_text_model_name == config.text_model_name
        shared_text_model = self.text_model if share_desc_backbone else None
        self.desc_fusion = IndependentDescriptionEncoder(
            desc_model_name=config.desc_text_model_name,
            audio_raw_dim=self.hidden_dim,
            visual_raw_dim=self.hidden_dim,
            desc_dim=config.desc_dim,
            dropout=config.dropout,
            hf_local_only=config.hf_local_only,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch_dtype,
            shared_text_model=shared_text_model,
        )
        self.desc_router_proj = nn.Sequential(
            nn.LayerNorm(config.desc_dim * 2),
            nn.Linear(config.desc_dim * 2, config.desc_router_proj_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        attn_kwargs = dict(
            dropout=config.dropout,
            mode=config.attnres_type,
            num_blocks=config.attnres_num_blocks,
            zero_init=config.attnres_zero_init,
            stable_residual=True,
            residual_init_alpha=config.attnres_init_alpha,
        )
        self.global_moe = EmotionGroupedMoEv4(
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
            desc_router_in_dim=config.desc_router_proj_dim,
            desc_logit_scale_init=config.desc_logit_scale_init,
            use_triplet_uniqueness=config.use_triplet_uniqueness,
            triplet_margin=config.triplet_margin,
            interaction_router_target_mode=config.moe_interaction_router_target_mode,
            interaction_router_temperature=config.moe_interaction_router_temperature,
            gate_temperature=config.moe_gate_temperature,
        )
        self.multimodal_attnres = ProjectedAttentionResidualFusion(
            [self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim],
            self.hidden_dim,
            **attn_kwargs,
        )

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.regression_head[-1].bias)

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self._set_frozen_backbones_eval()

    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _set_frozen_backbones_eval(self) -> None:
        self.text_model.eval()
        self.audio_model.eval()
        self.desc_fusion.desc_encoder.text_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_frozen_backbones_eval()
        return self

    def _encode_text(
        self,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.text_model.eval()
        with torch.no_grad():
            outputs = self.text_model(
                input_ids=text_inputs,
                attention_mask=text_mask,
                return_dict=True,
            )
            pooled_raw = _pool_text_output(outputs, text_mask)
        global_feat = self.text_pool_proj(pooled_raw.float())
        return global_feat, text_mask.bool()

    def _encode_audio(
        self,
        audio_inputs: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.audio_model.eval()
        audio_dtype = module_parameter_dtype(self.audio_model)
        with torch.no_grad():
            outputs = self.audio_model(
                input_values=audio_inputs.to(dtype=audio_dtype),
                attention_mask=audio_mask,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
        seq_len = hidden.size(1)
        if audio_mask is None:
            hidden_mask = torch.ones(
                hidden.size(0),
                seq_len,
                device=hidden.device,
                dtype=torch.bool,
            )
        else:
            raw_lengths = audio_mask.float().sum(dim=1)
            ratio = raw_lengths / max(1, audio_mask.size(1))
            hidden_lengths = torch.ceil(ratio * seq_len).long().clamp(min=1, max=seq_len)
            hidden_mask = sequence_mask_from_lengths(hidden_lengths, seq_len)
        audio_seq = self.audio_proj(hidden.float())
        audio_global = masked_mean_pool(audio_seq, hidden_mask)
        return audio_seq, audio_global, hidden_mask

    def _run_global_moe(
        self,
        text_global: torch.Tensor,
        audio_global: torch.Tensor,
        visual_global: torch.Tensor,
        desc_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        moe_aux_scale: float = 1.0,
        return_diagnostics: bool = False,
    ):
        global_cat = torch.cat([text_global, audio_global, visual_global], dim=-1)
        result = self.global_moe(
            global_cat,
            desc_features=desc_features,
            labels=labels,
            aux_scale=moe_aux_scale,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            return result
        out, scalar_loss, raw = result
        return out, scalar_loss, raw, {}

    def forward(
        self,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor,
        audio_inputs: torch.Tensor,
        audio_mask: torch.Tensor,
        visual_inputs: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        audio_desc_tokens: Optional[torch.Tensor] = None,
        audio_desc_masks: Optional[torch.Tensor] = None,
        audio_desc_valid: Optional[torch.Tensor] = None,
        visual_desc_tokens: Optional[torch.Tensor] = None,
        visual_desc_masks: Optional[torch.Tensor] = None,
        visual_desc_valid: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        moe_aux_scale: float = 1.0,
        moe_ablation_mode: str = "none",
        return_diagnostics: bool = False,
    ) -> dict:
        text_global, _ = self._encode_text(text_inputs, text_mask)
        audio_seq, audio_global, _ = self._encode_audio(audio_inputs, audio_mask)
        del audio_seq
        _, visual_global, _ = self.visual_encoder(visual_inputs, visual_mask)

        zero = audio_global.new_tensor(0.0)
        desc_audio, desc_visual, aux_losses = self.desc_fusion(
            audio_global,
            visual_global,
            audio_desc_tokens,
            audio_desc_masks,
            audio_desc_valid,
            visual_desc_tokens,
            visual_desc_masks,
            visual_desc_valid,
        )
        desc_proj = self.desc_router_proj(torch.cat([desc_audio, desc_visual], dim=-1))

        moe_feature, moe_loss, moe_raw_losses, moe_diagnostics = self._run_global_moe(
            text_global,
            audio_global,
            visual_global,
            desc_features=desc_proj,
            labels=labels,
            moe_aux_scale=moe_aux_scale,
            return_diagnostics=return_diagnostics,
        )

        if moe_ablation_mode == "none":
            pass
        elif moe_ablation_mode == "zero":
            moe_feature = torch.zeros_like(moe_feature)
        elif moe_ablation_mode == "mean":
            moe_feature = moe_feature.mean(dim=0, keepdim=True).expand_as(moe_feature)
        else:
            raise ValueError(f"Unknown moe_ablation_mode: {moe_ablation_mode}")

        attnres_sources = [text_global, audio_global, visual_global, moe_feature]
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
            diagnostics["final_source_names"] = ["text", "audio", "visual", "moe"]
            diagnostics["desc_to_interaction_only"] = torch.ones(
                (),
                device=regression_pred.device,
                dtype=zero.dtype,
            )
            outputs["_diagnostics"] = diagnostics
        return outputs


__all__ = ["PIERMoEForMER", "VisualFeatureEncoder"]

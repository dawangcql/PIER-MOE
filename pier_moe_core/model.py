from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import RobertaModel, WavLMModel

from .attention_residuals import ProjectedAttentionResidualFusion
from .config import PIERMoEConfig
from .description_fusion import MultiDescriptionFusion
from .emotion_grouped_moe import EmotionGroupedMoE
from .grouped_moe import GroupedMoE, LocalFeatureMoE


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

    def forward(self, visual_inputs: torch.Tensor, visual_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_inputs.dim() == 2:
            visual_inputs = visual_inputs.unsqueeze(1)
        if visual_mask is None:
            visual_mask = torch.ones(visual_inputs.size(0), visual_inputs.size(1), device=visual_inputs.device, dtype=torch.bool)
        visual_seq = self.frame_proj(visual_inputs.float())
        visual_global = masked_mean_pool(visual_seq, visual_mask)
        return visual_seq, visual_global, visual_mask.bool()


class PIERMoEForMER(nn.Module):
    """
    Simplified PIERMoE model for MOSI/MOSEI.

    Pipeline: Encode → Context → Description → GroupedMoE → AttnRes Fusion → Regression.

    Removed vs. the full version: Cross-Modal Decoder and Label-Guided Prediction.
    """

    def __init__(self, config: PIERMoEConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.use_context = config.use_context
        self.use_visual = config.use_visual
        self.use_desc = config.use_desc
        self.use_grouped_moe = config.use_grouped_moe
        self.use_local_moe = config.use_grouped_moe and config.use_local_moe

        self.roberta_model = RobertaModel.from_pretrained(config.text_model_name, local_files_only=config.hf_local_only)
        self.wavlm_model = WavLMModel.from_pretrained(config.audio_model_name_or_path, local_files_only=config.hf_local_only)

        text_hidden = self.roberta_model.config.hidden_size
        audio_hidden = self.wavlm_model.config.hidden_size
        self.text_token_proj = nn.Sequential(nn.LayerNorm(text_hidden), nn.Linear(text_hidden, self.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout))
        self.text_pool_proj = nn.Sequential(nn.LayerNorm(text_hidden), nn.Linear(text_hidden, self.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout))
        self.audio_proj = nn.Sequential(nn.LayerNorm(audio_hidden), nn.Linear(audio_hidden, self.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout))
        self.visual_encoder = VisualFeatureEncoder(config.visual_dim, self.hidden_dim, config.dropout) if self.use_visual else None

        self.desc_fusion = (
            MultiDescriptionFusion(
                desc_model_name=config.desc_text_model_name,
                audio_raw_dim=self.hidden_dim,
                visual_raw_dim=self.hidden_dim,
                desc_dim=config.desc_dim,
                openface_dim=config.openface_dim,
                out_dim=self.hidden_dim,
                freeze_text_encoder=config.freeze_desc_encoder,
                dropout=config.dropout,
                hf_local_only=config.hf_local_only,
            )
            if self.use_desc
            else None
        )

        attn_kwargs = dict(
            dropout=config.dropout,
            mode=config.attnres_type,
            num_blocks=config.attnres_num_blocks,
            zero_init=config.attnres_zero_init,
            stable_residual=True,
            residual_init_alpha=config.attnres_init_alpha,
        )
        self.text_context_attnres = ProjectedAttentionResidualFusion([self.hidden_dim, self.hidden_dim, self.hidden_dim * 2], self.hidden_dim, **attn_kwargs)
        self.audio_context_attnres = ProjectedAttentionResidualFusion([self.hidden_dim, self.hidden_dim, self.hidden_dim * 2], self.hidden_dim, **attn_kwargs)
        self.visual_context_attnres = ProjectedAttentionResidualFusion([self.hidden_dim, self.hidden_dim, self.hidden_dim * 2], self.hidden_dim, **attn_kwargs)
        self.audio_desc_attnres = ProjectedAttentionResidualFusion([self.hidden_dim, self.hidden_dim], self.hidden_dim, **attn_kwargs)
        self.visual_desc_attnres = ProjectedAttentionResidualFusion([self.hidden_dim, self.hidden_dim], self.hidden_dim, **attn_kwargs)

        if self.use_grouped_moe:
            if config.use_emotion_moe:
                self.global_moe = EmotionGroupedMoE(
                    input_size=self.hidden_dim * 3,
                    output_size=self.hidden_dim,
                    hidden_size=self.hidden_dim,
                    n_modalities=3,
                    dropout=config.dropout,
                    shared_alpha_init=config.moe_shared_alpha_init,
                    polarity_threshold=config.moe_polarity_threshold,
                    mask_strategy=config.moe_mask_strategy,
                    loss_weights={
                        "polarity_guide": config.moe_polarity_guide_weight,
                        "interaction_pid": config.moe_interaction_pid_weight,
                        "balance": config.moe_balance_weight,
                    },
                )
            else:
                self.global_moe = GroupedMoE(
                    input_size=self.hidden_dim * 3,
                    output_size=self.hidden_dim,
                    hidden_size=self.hidden_dim,
                    n=3,
                    d=self.hidden_dim,
                    num_experts_per_type=config.moe_num_experts_per_type,
                    k=config.moe_k,
                )
            if self.use_local_moe:
                self.text_local_moe = LocalFeatureMoE(self.hidden_dim, self.hidden_dim, config.local_gran_text, config.moe_num_experts_per_type, config.moe_k, config.dropout)
                self.audio_local_moe = LocalFeatureMoE(self.hidden_dim, self.hidden_dim, config.local_gran_audio, config.moe_num_experts_per_type, config.moe_k, config.dropout)
                self.visual_local_moe = LocalFeatureMoE(self.hidden_dim, self.hidden_dim, config.local_gran_visual, config.moe_num_experts_per_type, config.moe_k, config.dropout)
                self.local_moe = GroupedMoE(
                    input_size=self.hidden_dim * 3,
                    output_size=self.hidden_dim,
                    hidden_size=self.hidden_dim,
                    n=3,
                    d=self.hidden_dim,
                    num_experts_per_type=config.moe_num_experts_per_type,
                    k=config.moe_k,
                )
                self.moe_attnres = ProjectedAttentionResidualFusion([self.hidden_dim, self.hidden_dim], self.hidden_dim, **attn_kwargs)
            else:
                self.moe_attnres = None
        else:
            self.global_moe = None
            self.moe_attnres = None

        final_sources = [self.hidden_dim, self.hidden_dim, self.hidden_dim]
        if self.use_grouped_moe:
            final_sources.append(self.hidden_dim)
        self.multimodal_attnres = ProjectedAttentionResidualFusion(final_sources, self.hidden_dim, **attn_kwargs)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.regression_head[-1].bias)

        self.T_output_layers = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.hidden_dim, 1))
        self.A_output_layers = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.hidden_dim, 1))
        self.V_output_layers = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.hidden_dim, 1))

    def _encode_text(self, text_inputs: torch.Tensor, text_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.roberta_model(input_ids=text_inputs, attention_mask=text_mask, return_dict=True)
        token_seq = self.text_token_proj(outputs.last_hidden_state)
        if outputs.pooler_output is not None:
            pooled_raw = outputs.pooler_output
        else:
            pooled_raw = masked_mean_pool(outputs.last_hidden_state, text_mask)
        global_feat = self.text_pool_proj(pooled_raw)
        return token_seq, global_feat, text_mask.bool()

    def _encode_audio(self, audio_inputs: torch.Tensor, audio_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.wavlm_model(input_values=audio_inputs, attention_mask=audio_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        seq_len = hidden.size(1)
        if audio_mask is None:
            hidden_mask = torch.ones(hidden.size(0), seq_len, device=hidden.device, dtype=torch.bool)
        else:
            raw_lengths = audio_mask.float().sum(dim=1)
            ratio = raw_lengths / max(1, audio_mask.size(1))
            hidden_lengths = torch.ceil(ratio * seq_len).long().clamp(min=1, max=seq_len)
            hidden_mask = sequence_mask_from_lengths(hidden_lengths, seq_len)
        audio_seq = self.audio_proj(hidden)
        audio_global = masked_mean_pool(audio_seq, hidden_mask)
        return audio_seq, audio_global, hidden_mask

    def _contextualize(
        self,
        current: torch.Tensor,
        context: Optional[torch.Tensor],
        context_valid: Optional[torch.Tensor],
        module: ProjectedAttentionResidualFusion,
    ) -> torch.Tensor:
        if context is None or context_valid is None:
            return current
        merged = module([current, context, torch.cat([current, context], dim=-1)])
        return torch.where(context_valid.view(-1, 1).to(device=current.device), merged, current)

    def _encode_optional_context(
        self,
        text_context_inputs,
        text_context_mask,
        audio_context_inputs,
        audio_context_mask,
        visual_context_inputs,
        visual_context_mask,
    ):
        text_context = text_context_valid = None
        audio_context = audio_context_valid = None
        visual_context = visual_context_valid = None
        if self.use_context and text_context_inputs is not None:
            _, text_context, _ = self._encode_text(text_context_inputs, text_context_mask)
            text_context_valid = text_context_mask.sum(dim=1) > 2
        if self.use_context and audio_context_inputs is not None:
            _, audio_context, _ = self._encode_audio(audio_context_inputs, audio_context_mask)
            audio_context_valid = audio_context_mask.sum(dim=1) > 0
        if self.use_context and self.use_visual and visual_context_inputs is not None:
            _, visual_context, _ = self.visual_encoder(visual_context_inputs, visual_context_mask)
            visual_context_valid = visual_context_mask.sum(dim=1) > 0
        return text_context, text_context_valid, audio_context, audio_context_valid, visual_context, visual_context_valid

    def _run_global_moe(
        self,
        global_cat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Run the global MoE and normalise the return tuple.

        Returns (output, scalar_loss, raw_loss_dict). The dict is empty for
        the legacy GroupedMoE.
        """
        if isinstance(self.global_moe, EmotionGroupedMoE):
            out, scalar_loss, raw = self.global_moe(global_cat, labels=labels)
            return out, scalar_loss, raw
        out, scalar_loss = self.global_moe(global_cat)
        return out, scalar_loss, {}

    def _build_moe_feature(
        self,
        text_global: torch.Tensor,
        audio_global: torch.Tensor,
        visual_global: torch.Tensor,
        text_seq: torch.Tensor,
        text_mask: torch.Tensor,
        audio_seq: torch.Tensor,
        audio_mask: torch.Tensor,
        visual_seq: torch.Tensor,
        visual_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        zero_loss = text_global.new_tensor(0.0)
        if not self.use_grouped_moe:
            return None, zero_loss, {}

        global_cat = torch.cat([text_global, audio_global, visual_global], dim=-1)
        global_moe, moe_loss, moe_raw = self._run_global_moe(global_cat, labels=labels)
        if not self.use_local_moe:
            return global_moe, moe_loss, moe_raw

        text_local, text_loss = self.text_local_moe(text_seq, text_mask)
        audio_local, audio_loss = self.audio_local_moe(audio_seq, audio_mask)
        visual_local, visual_loss = self.visual_local_moe(visual_seq, visual_mask)
        local_cat = torch.cat([text_local, audio_local, visual_local], dim=-1)
        local_moe, local_loss = self.local_moe(local_cat)
        moe_fused = self.moe_attnres([global_moe, local_moe])
        total_loss = moe_loss + text_loss + audio_loss + visual_loss + local_loss
        return moe_fused, total_loss, moe_raw

    def forward(
        self,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor,
        audio_inputs: torch.Tensor,
        audio_mask: torch.Tensor,
        visual_inputs: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        text_context_inputs: Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
        audio_context_inputs: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        visual_context_inputs: Optional[torch.Tensor] = None,
        visual_context_mask: Optional[torch.Tensor] = None,
        audio_desc_tokens: Optional[torch.Tensor] = None,
        audio_desc_masks: Optional[torch.Tensor] = None,
        audio_desc_valid: Optional[torch.Tensor] = None,
        visual_desc_tokens: Optional[torch.Tensor] = None,
        visual_desc_masks: Optional[torch.Tensor] = None,
        visual_desc_valid: Optional[torch.Tensor] = None,
        openface_inputs: Optional[torch.Tensor] = None,
        openface_valid: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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

        # ── Context ──
        ctx = self._encode_optional_context(
            text_context_inputs, text_context_mask,
            audio_context_inputs, audio_context_mask,
            visual_context_inputs, visual_context_mask,
        )
        text_ctx, text_ctx_valid, audio_ctx, audio_ctx_valid, visual_ctx, visual_ctx_valid = ctx
        text_global = self._contextualize(text_global, text_ctx, text_ctx_valid, self.text_context_attnres)
        audio_global = self._contextualize(audio_global, audio_ctx, audio_ctx_valid, self.audio_context_attnres)
        visual_global = self._contextualize(visual_global, visual_ctx, visual_ctx_valid, self.visual_context_attnres)

        # ── Description fusion ──
        if self.use_desc:
            audio_desc, visual_desc, aux_losses = self.desc_fusion(
                audio_global, visual_global,
                audio_desc_tokens, audio_desc_masks, audio_desc_valid,
                visual_desc_tokens, visual_desc_masks, visual_desc_valid,
                openface_inputs, openface_valid,
            )
            audio_global = self.audio_desc_attnres([audio_global, audio_desc])
            visual_global = self.visual_desc_attnres([visual_global, visual_desc])
        else:
            aux_losses = {
                "audio_align": audio_global.new_tensor(0.0),
                "visual_align": audio_global.new_tensor(0.0),
                "visual_openface_align": audio_global.new_tensor(0.0),
            }

        # ── Grouped MoE ──
        moe_feature, moe_loss, moe_raw_losses = self._build_moe_feature(
            text_global, audio_global, visual_global,
            text_seq, text_seq_mask,
            audio_seq, audio_seq_mask,
            visual_seq, visual_seq_mask,
            labels=labels,
        )

        # ── Multimodal fusion ──
        attnres_sources = [text_global, audio_global, visual_global]
        if moe_feature is not None:
            attnres_sources.append(moe_feature)
        multimodal_feature = self.multimodal_attnres(attnres_sources)

        # ── Prediction ──
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
        return outputs

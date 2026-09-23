from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


def _pool_text_output(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    hidden = outputs.last_hidden_state
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


class DescriptionTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int = 256,
        freeze: bool = True,
        dropout: float = 0.1,
        hf_local_only: bool = True,
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name, local_files_only=hf_local_only)
        self.freeze = freeze
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
        hidden = self.text_model.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = _pool_text_output(outputs, attention_mask)
        return self.proj(pooled)


class MaskedGatedFusion(nn.Module):
    def __init__(self, raw_dim: int, aux_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.raw_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(raw_dim, out_dim), nn.ReLU())
        self.aux_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(aux_dim, out_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())

    def forward(self, raw_feat: torch.Tensor, aux_feat: torch.Tensor, aux_valid: torch.Tensor) -> torch.Tensor:
        raw_h = self.raw_proj(raw_feat)
        aux_h = self.aux_proj(aux_feat)
        gate = self.gate(torch.cat([raw_h, aux_h], dim=-1))
        fused = gate * raw_h + (1.0 - gate) * aux_h
        aux_valid = aux_valid.float().view(-1, 1)
        return aux_valid * fused + (1.0 - aux_valid) * raw_h


def masked_cosine_align_loss(raw_proj: torch.Tensor, aux_feat: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = valid_mask.float().view(-1)
    raw_proj = F.normalize(raw_proj, dim=-1)
    aux_feat = F.normalize(aux_feat, dim=-1)
    loss = 1.0 - (raw_proj * aux_feat).sum(dim=-1)
    return (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)


class MultiDescriptionFusion(nn.Module):
    """Fuse MLLM-generated audio and visual descriptions into raw MMML features."""

    def __init__(
        self,
        desc_model_name: str,
        audio_raw_dim: int,
        visual_raw_dim: int,
        desc_dim: int,
        openface_dim: int,
        out_dim: int,
        freeze_text_encoder: bool = True,
        dropout: float = 0.1,
        hf_local_only: bool = True,
    ):
        super().__init__()
        del openface_dim
        self.desc_encoder = DescriptionTextEncoder(
            model_name=desc_model_name,
            out_dim=desc_dim,
            freeze=freeze_text_encoder,
            dropout=dropout,
            hf_local_only=hf_local_only,
        )
        self.audio_fuser = MaskedGatedFusion(audio_raw_dim, desc_dim, out_dim, dropout=dropout)
        self.visual_fuser = MaskedGatedFusion(visual_raw_dim, desc_dim, out_dim, dropout=dropout)
        self.audio_align_proj = nn.Linear(audio_raw_dim, desc_dim)
        self.visual_align_proj = nn.Linear(visual_raw_dim, desc_dim)

    def forward(
        self,
        audio_raw_feat: torch.Tensor,
        visual_raw_feat: torch.Tensor,
        audio_desc_tokens: Optional[torch.Tensor],
        audio_desc_masks: Optional[torch.Tensor],
        audio_desc_valid: Optional[torch.Tensor],
        visual_desc_tokens: Optional[torch.Tensor],
        visual_desc_masks: Optional[torch.Tensor],
        visual_desc_valid: Optional[torch.Tensor],
        openface_inputs: Optional[torch.Tensor] = None,
        openface_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        del openface_inputs, openface_valid
        zero = audio_raw_feat.new_tensor(0.0)
        if audio_desc_tokens is None or visual_desc_tokens is None:
            return self.audio_fuser.raw_proj(audio_raw_feat), self.visual_fuser.raw_proj(visual_raw_feat), {
                "audio_align": zero,
                "visual_align": zero,
                "visual_openface_align": zero,
            }

        audio_desc_feat = self.desc_encoder(audio_desc_tokens, audio_desc_masks)
        visual_desc_feat = self.desc_encoder(visual_desc_tokens, visual_desc_masks)
        audio_desc_valid = torch.ones(audio_raw_feat.size(0), device=audio_raw_feat.device) if audio_desc_valid is None else audio_desc_valid
        visual_desc_valid = torch.ones(visual_raw_feat.size(0), device=visual_raw_feat.device) if visual_desc_valid is None else visual_desc_valid

        audio_fused = self.audio_fuser(audio_raw_feat, audio_desc_feat, audio_desc_valid)
        visual_fused = self.visual_fuser(visual_raw_feat, visual_desc_feat, visual_desc_valid)
        losses = {
            "audio_align": masked_cosine_align_loss(self.audio_align_proj(audio_raw_feat), audio_desc_feat, audio_desc_valid),
            "visual_align": masked_cosine_align_loss(self.visual_align_proj(visual_raw_feat), visual_desc_feat, visual_desc_valid),
            "visual_openface_align": zero,
        }
        return audio_fused, visual_fused, losses

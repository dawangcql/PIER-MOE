from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


def _resolve_torch_dtype(value):
    if value in (None, "none"):
        return None
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    return value


def _model_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError("Cannot infer hidden size from model.config.")


def _pool_text_output(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    hidden = outputs.last_hidden_state
    if attention_mask is None:
        return hidden[:, -1, :]
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, lengths]


class DescriptionTextEncoder(nn.Module):
    """Frozen Baichuan encoder plus a trainable projection to ``desc_dim``."""

    def __init__(
        self,
        model_name: str,
        out_dim: int = 256,
        dropout: float = 0.1,
        hf_local_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype="auto",
        shared_text_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        if shared_text_model is None:
            kwargs = {
                "local_files_only": hf_local_only,
                "trust_remote_code": trust_remote_code,
            }
            torch_dtype = _resolve_torch_dtype(torch_dtype)
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            self.text_model = AutoModel.from_pretrained(model_name, **kwargs)
        else:
            self.text_model = shared_text_model
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
        hidden = _model_hidden_size(self.text_model)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.text_model.eval()
        with torch.no_grad():
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled = _pool_text_output(outputs, attention_mask)
        return self.proj(pooled.float())


def _masked_cosine_align_loss(
    raw_proj: torch.Tensor,
    aux_feat: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid_mask = valid_mask.float().view(-1)
    raw_proj = F.normalize(raw_proj, dim=-1)
    aux_feat = F.normalize(aux_feat, dim=-1)
    loss = 1.0 - (raw_proj * aux_feat).sum(dim=-1)
    return (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)


class IndependentDescriptionEncoder(nn.Module):
    def __init__(
        self,
        desc_model_name: str,
        audio_raw_dim: int,
        visual_raw_dim: int,
        desc_dim: int,
        dropout: float = 0.1,
        hf_local_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype="auto",
        shared_text_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.desc_encoder = DescriptionTextEncoder(
            model_name=desc_model_name,
            out_dim=desc_dim,
            dropout=dropout,
            hf_local_only=hf_local_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            shared_text_model=shared_text_model,
        )
        self.audio_align_proj = nn.Linear(audio_raw_dim, desc_dim)
        self.visual_align_proj = nn.Linear(visual_raw_dim, desc_dim)
        self.desc_dim = desc_dim

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
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        zero = audio_raw_feat.new_tensor(0.0)
        batch = audio_raw_feat.size(0)
        if audio_desc_tokens is None or visual_desc_tokens is None:
            empty = audio_raw_feat.new_zeros(batch, self.desc_dim)
            return empty, empty, {
                "audio_align": zero,
                "visual_align": zero,
                "visual_openface_align": zero,
            }

        audio_desc_feat = self.desc_encoder(audio_desc_tokens, audio_desc_masks)
        visual_desc_feat = self.desc_encoder(visual_desc_tokens, visual_desc_masks)

        if audio_desc_valid is None:
            audio_desc_valid = torch.ones(batch, device=audio_raw_feat.device)
        if visual_desc_valid is None:
            visual_desc_valid = torch.ones(batch, device=audio_raw_feat.device)

        a_mask = audio_desc_valid.float().view(-1, 1)
        v_mask = visual_desc_valid.float().view(-1, 1)
        audio_desc_out = a_mask * audio_desc_feat
        visual_desc_out = v_mask * visual_desc_feat

        losses = {
            "audio_align": _masked_cosine_align_loss(
                self.audio_align_proj(audio_raw_feat),
                audio_desc_feat,
                audio_desc_valid,
            ),
            "visual_align": _masked_cosine_align_loss(
                self.visual_align_proj(visual_raw_feat),
                visual_desc_feat,
                visual_desc_valid,
            ),
            "visual_openface_align": zero,
        }
        return audio_desc_out, visual_desc_out, losses


__all__ = ["IndependentDescriptionEncoder", "DescriptionTextEncoder"]

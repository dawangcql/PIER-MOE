from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer

NO_AUDIO_DESC = "[NO_AUDIO_DESC]"
NO_VISUAL_DESC = "[NO_VISUAL_DESC]"
VALID_AUDIO_STATUSES = {"ok"}
VALID_VISUAL_STATUSES = {"ok", "ok_from_openface_rule"}


def load_desc_jsonl(desc_jsonl_path: str) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not desc_jsonl_path or not os.path.exists(desc_jsonl_path):
        return records
    with open(desc_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records[obj["sample_id"]] = obj
    return records


class DatasetMosiWithDesc(Dataset):
    def __init__(
        self,
        csv_path: str,
        audio_directory: str,
        mode: str,
        root_dir: str,
        desc_jsonl_path: Optional[str] = None,
        text_context_length: int = 2,
        audio_context_length: int = 1,
        visual_context_length: int = 1,
        use_visual: bool = False,
        visual_directory: Optional[str] = None,
        visual_num_frames: int = 16,
        visual_dim: int = 768,
        text_model_name: str = "/root/autodl-tmp/models/roberta-large",
        desc_text_model_name: str = "/root/autodl-tmp/models/roberta-large",
        audio_feature_extractor_name: str = "/root/autodl-tmp/models/wavlm-large",
        openface_dim: int = 63,
        hf_local_only: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)
        invalid_files = ["3aIQUQgawaI/12.wav", "94ULum9MYX0/2.wav", "mRnEJOLkhp8/24.wav", "aE-X_QdDaqQ/3.wav", "94ULum9MYX0/11.wav", "mRnEJOLkhp8/26.wav"]
        for f in invalid_files:
            video_id = f.split("/")[0]
            clip_id = f.split("/")[1].split(".")[0]
            df = df[~((df["video_id"] == video_id) & (df["clip_id"] == int(clip_id)))]
        df = df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index(drop=True)

        self.targets_M = df["label"]
        df["text"] = df["text"].str[0] + df["text"].str[1::].apply(lambda x: x.lower())
        self.texts = df["text"]
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=hf_local_only)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(desc_text_model_name, local_files_only=hf_local_only)

        self.audio_file_paths = []
        self.visual_file_paths = []
        self.sample_ids = []
        self.root_dir = root_dir
        self.openface_dim = openface_dim
        for i in range(len(df)):
            stem_name = str(df["video_id"][i]) + "/" + str(df["clip_id"][i])
            self.sample_ids.append(stem_name)
            self.audio_file_paths.append(os.path.join(audio_directory, stem_name + ".wav"))
            self.visual_file_paths.append(os.path.join(visual_directory, stem_name + ".npz") if visual_directory else "")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_feature_extractor_name, local_files_only=hf_local_only)
        self.video_id = df["video_id"]
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length
        self.visual_context_length = visual_context_length
        self.use_visual = use_visual
        self.visual_num_frames = visual_num_frames
        self.visual_dim = visual_dim
        self.desc_records = load_desc_jsonl(desc_jsonl_path or os.path.join(root_dir, "mllm_desc.jsonl"))

    def _empty_visual(self):
        return (
            torch.zeros((self.visual_num_frames, self.visual_dim), dtype=torch.float32),
            torch.zeros((self.visual_num_frames,), dtype=torch.long),
        )

    def _normalize_visual_shape(self, visual_features, visual_masks):
        if visual_features.ndim == 1:
            visual_features = visual_features.unsqueeze(0)
        if visual_masks.ndim == 0:
            visual_masks = visual_masks.unsqueeze(0)
        if visual_features.shape[-1] != self.visual_dim:
            if visual_features.shape[-1] > self.visual_dim:
                visual_features = visual_features[:, : self.visual_dim]
            else:
                pad = torch.zeros((visual_features.shape[0], self.visual_dim - visual_features.shape[-1]), dtype=visual_features.dtype)
                visual_features = torch.cat((visual_features, pad), dim=-1)
        num_frames = visual_features.shape[0]
        if num_frames > self.visual_num_frames:
            visual_features = visual_features[: self.visual_num_frames]
            visual_masks = visual_masks[: self.visual_num_frames]
        elif num_frames < self.visual_num_frames:
            pad_feat = torch.zeros((self.visual_num_frames - num_frames, self.visual_dim), dtype=visual_features.dtype)
            pad_mask = torch.zeros((self.visual_num_frames - num_frames,), dtype=visual_masks.dtype)
            visual_features = torch.cat((visual_features, pad_feat), dim=0)
            visual_masks = torch.cat((visual_masks, pad_mask), dim=0)
        return visual_features, visual_masks

    def _load_visual(self, index):
        if not self.use_visual:
            return self._empty_visual()
        visual_path = self.visual_file_paths[index]
        if not visual_path or not os.path.exists(visual_path):
            return self._empty_visual()
        try:
            with np.load(visual_path) as data:
                if "feat" in data:
                    visual_features = data["feat"]
                elif "features" in data:
                    visual_features = data["features"]
                else:
                    return self._empty_visual()
                if "mask" in data:
                    visual_masks = data["mask"]
                else:
                    visual_masks = np.ones((visual_features.shape[0],), dtype=np.int64)
        except Exception:
            return self._empty_visual()
        visual_features = torch.tensor(visual_features, dtype=torch.float32)
        visual_masks = torch.tensor(visual_masks, dtype=torch.long)
        return self._normalize_visual_shape(visual_features, visual_masks)

    def _encode_desc(self, text: str):
        tokenized = self.desc_tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return (
            torch.tensor(tokenized["input_ids"], dtype=torch.long),
            torch.tensor(tokenized["attention_mask"], dtype=torch.long),
        )

    def _empty_openface(self):
        return (
            torch.zeros((self.openface_dim,), dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

    def __getitem__(self, index):
        text = str(self.texts[index])

        text_context = ""
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            context = str(self.texts[index - i])
            text_context = context + "</s>" + text_context

        tokenized_text = self.tokenizer(text, max_length=96, padding="max_length", truncation=True, add_special_tokens=True, return_attention_mask=True)
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(text_context, max_length=96, padding="max_length", truncation=True, add_special_tokens=True, return_attention_mask=True)

        sound, _ = torchaudio.load(self.audio_file_paths[index])
        sound_data = torch.mean(sound, dim=0, keepdim=False)

        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            context, _ = torchaudio.load(self.audio_file_paths[index - i])
            context_data = torch.mean(context, dim=0, keepdim=False)
            audio_context = torch.cat((context_data, audio_context), 0)

        features = self.feature_extractor(sound_data, sampling_rate=16000, max_length=96000, return_attention_mask=True, truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features["input_values"]), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features["attention_mask"]), dtype=torch.long).squeeze()

        if len(audio_context) == 0:
            # 明确指定 features 为 float32，masks 为 long
            audio_context_features = torch.zeros(96000, dtype=torch.float32)
            audio_context_masks = torch.zeros(96000, dtype=torch.long)
        else:
            features = self.feature_extractor(audio_context, sampling_rate=16000, max_length=96000, return_attention_mask=True, truncation=True, padding="max_length")
            audio_context_features = torch.tensor(np.array(features["input_values"]), dtype=torch.float32).squeeze()
            audio_context_masks = torch.tensor(np.array(features["attention_mask"]), dtype=torch.long).squeeze()

        visual_features, visual_masks = self._load_visual(index)
        visual_context_chunks = []
        for i in range(1, self.visual_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            context_feat, context_mask = self._load_visual(index - i)
            valid_context_feat = context_feat[context_mask.bool()]
            if valid_context_feat.numel() > 0:
                visual_context_chunks.insert(0, valid_context_feat)
        if len(visual_context_chunks) > 0:
            visual_context_all = torch.cat(visual_context_chunks, dim=0)
            visual_context_mask_all = torch.ones((visual_context_all.shape[0],), dtype=torch.long)
            visual_context_features, visual_context_masks = self._normalize_visual_shape(visual_context_all, visual_context_mask_all)
        else:
            visual_context_features, visual_context_masks = self._empty_visual()

        sample_id = self.sample_ids[index]
        desc_item = self.desc_records.get(sample_id, {})
        audio_desc = str(desc_item.get("audio_desc", desc_item.get("audio_description", "")) or "")
        visual_desc = str(desc_item.get("visual_desc", desc_item.get("visual_description", "")) or "")
        audio_desc_valid = float(desc_item.get("audio_status", "") in VALID_AUDIO_STATUSES)
        visual_desc_valid = float(desc_item.get("visual_status", "") in VALID_VISUAL_STATUSES)
        if audio_desc_valid == 0.0:
            audio_desc = NO_AUDIO_DESC
        if visual_desc_valid == 0.0:
            visual_desc = NO_VISUAL_DESC
        audio_desc_tokens, audio_desc_masks = self._encode_desc(audio_desc)
        visual_desc_tokens, visual_desc_masks = self._encode_desc(visual_desc)
        openface_feature, openface_valid = self._empty_openface()

        return {
            "sample_id": sample_id,
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "audio_context_inputs": audio_context_features,
            "audio_context_masks": audio_context_masks,
            "visual_inputs": visual_features,
            "visual_masks": visual_masks,
            "visual_context_inputs": visual_context_features,
            "visual_context_masks": visual_context_masks,
            "audio_desc_tokens": audio_desc_tokens,
            "audio_desc_masks": audio_desc_masks,
            "audio_desc_valid": torch.tensor(audio_desc_valid, dtype=torch.float32),
            "visual_desc_tokens": visual_desc_tokens,
            "visual_desc_masks": visual_desc_masks,
            "visual_desc_valid": torch.tensor(visual_desc_valid, dtype=torch.float32),
            "openface_inputs": openface_feature,
            "openface_valid": openface_valid,
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.targets_M)


def build_loaders_with_desc(
    root_dir: str,
    batch_size: int,
    desc_jsonl_path: Optional[str] = None,
    text_context_length: int = 2,
    audio_context_length: int = 1,
    visual_context_length: int = 1,
    use_visual: bool = True,
    visual_num_frames: int = 16,
    visual_dim: int = 768,
    text_model_name: str = "/root/autodl-tmp/models/roberta-large",
    desc_text_model_name: str = "/root/autodl-tmp/models/roberta-large",
    audio_feature_extractor_name: str = "/root/autodl-tmp/models/wavlm-large",
    hf_local_only: bool = True,
    seed: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    csv_path = os.path.join(root_dir, "label.csv")
    audio_file_path = os.path.join(root_dir, "wav")
    visual_file_path = os.path.join(root_dir, "visual")
    train_data = DatasetMosiWithDesc(
        csv_path,
        audio_file_path,
        "train",
        root_dir,
        desc_jsonl_path,
        text_context_length,
        audio_context_length,
        visual_context_length,
        use_visual,
        visual_file_path,
        visual_num_frames,
        visual_dim,
        text_model_name,
        desc_text_model_name,
        audio_feature_extractor_name,
        hf_local_only=hf_local_only,
    )
    test_data = DatasetMosiWithDesc(
        csv_path,
        audio_file_path,
        "test",
        root_dir,
        desc_jsonl_path,
        text_context_length,
        audio_context_length,
        visual_context_length,
        use_visual,
        visual_file_path,
        visual_num_frames,
        visual_dim,
        text_model_name,
        desc_text_model_name,
        audio_feature_extractor_name,
        hf_local_only=hf_local_only,
    )
    val_data = DatasetMosiWithDesc(
        csv_path,
        audio_file_path,
        "valid",
        root_dir,
        desc_jsonl_path,
        text_context_length,
        audio_context_length,
        visual_context_length,
        use_visual,
        visual_file_path,
        visual_num_frames,
        visual_dim,
        text_model_name,
        desc_text_model_name,
        audio_feature_extractor_name,
        hf_local_only=hf_local_only,
    )

    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, **loader_args)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **loader_args)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_args)
    return train_loader, test_loader, val_loader

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    from .config import PIERMoEConfig
    from .model import PIERMoEForMER
except ImportError:
    from config import PIERMoEConfig
    from model import PIERMoEForMER


class TrainLogger:
    def __init__(self, log_dir: str, run_name: str, hyperparams: dict | None = None):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(self.log_dir, exist_ok=True)
        self.epoch_path = os.path.join(self.log_dir, f"{self.run_name}_epochs.jsonl")
        self.summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")
        if hyperparams is not None:
            with open(os.path.join(self.log_dir, f"{self.run_name}_hparams.json"), "w", encoding="utf-8") as f:
                json.dump(hyperparams, f, indent=2, ensure_ascii=False)

    def log_epoch(self, record: dict) -> None:
        with open(self.epoch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_summary(self, record: dict) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def dataset_root(dataset_name: str, data_root: str) -> str:
    if data_root:
        return data_root
    if dataset_name.lower() == "mosi":
        return "data/MOSI"
    if dataset_name.lower() == "mosei":
        return "data/MOSEI"
    raise ValueError("PIERMoE expects MOSI or MOSEI.")


def build_optimizer(args, model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )


def build_scheduler(args, optimizer, total_steps: int):
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_model_stats(model: nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
        "params_size_mb": param_bytes / (1024 ** 2),
        "trainable_params_size_mb": trainable_param_bytes / (1024 ** 2),
        "buffers_size_mb": buffer_bytes / (1024 ** 2),
        "model_size_mb": total_bytes / (1024 ** 2),
    }


def trainable_state_dict(model: nn.Module) -> dict:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    return {
        name: value.detach().cpu()
        for name, value in state.items()
        if name in trainable_names
    }


def compute_aux_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - 1) / float(warmup_epochs))))


def dataset_polarity_thresholds(dataset: str) -> tuple[float, float]:
    name = dataset.lower()
    if name == "mosi":
        return 0.2, 1.0
    if name == "mosei":
        return 0.33, 1.0
    raise ValueError(f"Unknown dataset: {dataset}")


def resolve_polarity_thresholds(args) -> tuple[float, float]:
    default_neutral, default_strong = dataset_polarity_thresholds(args.dataset)
    neutral = default_neutral if args.moe_polarity_threshold < 0 else args.moe_polarity_threshold
    strong = (
        default_strong
        if args.moe_polarity_strong_threshold < 0
        else args.moe_polarity_strong_threshold
    )
    if strong <= neutral:
        raise ValueError(
            "moe_polarity_strong_threshold must be greater than moe_polarity_threshold."
        )
    return neutral, strong


class Trainer:
    def __init__(self, args, metrics, device: torch.device):
        self.args = args
        self.device = device
        self.metrics = metrics
        self.l1 = nn.L1Loss()

    def forward_batch(
        self,
        model: PIERMoEForMER,
        batch: dict,
        labels=None,
        moe_aux_scale: float = 1.0,
    ) -> dict:
        kwargs = {
            "text_inputs": batch["text_tokens"].to(self.device),
            "text_mask": batch["text_masks"].to(self.device),
            "audio_inputs": batch["audio_inputs"].to(self.device),
            "audio_mask": batch["audio_masks"].to(self.device),
            "visual_inputs": batch["visual_inputs"].to(self.device),
            "visual_mask": batch["visual_masks"].to(self.device),
            "audio_desc_tokens": batch.get("audio_desc_tokens", None).to(self.device)
            if "audio_desc_tokens" in batch
            else None,
            "audio_desc_masks": batch.get("audio_desc_masks", None).to(self.device)
            if "audio_desc_masks" in batch
            else None,
            "audio_desc_valid": batch.get("audio_desc_valid", None).to(self.device)
            if "audio_desc_valid" in batch
            else None,
            "visual_desc_tokens": batch.get("visual_desc_tokens", None).to(self.device)
            if "visual_desc_tokens" in batch
            else None,
            "visual_desc_masks": batch.get("visual_desc_masks", None).to(self.device)
            if "visual_desc_masks" in batch
            else None,
            "visual_desc_valid": batch.get("visual_desc_valid", None).to(self.device)
            if "visual_desc_valid" in batch
            else None,
            "labels": labels,
            "moe_aux_scale": moe_aux_scale,
        }
        return model(**kwargs)

    def compute_total_loss(self, outputs: dict, targets: torch.Tensor):
        mae = self.l1(outputs["M"], targets)
        total = mae
        if self.args.unimodal_aux_weight > 0:
            missing = [key for key in ("T", "A", "V") if key not in outputs]
            if missing:
                raise RuntimeError(f"Missing auxiliary outputs: {missing}.")
            unimodal_terms = []
            weights = []
            for key, weight in (
                ("T", self.args.unimodal_aux_text_weight),
                ("A", self.args.unimodal_aux_audio_weight),
                ("V", self.args.unimodal_aux_visual_weight),
            ):
                if weight > 0:
                    unimodal_terms.append(weight * self.l1(outputs[key], targets))
                    weights.append(weight)
            if unimodal_terms:
                total = total + self.args.unimodal_aux_weight * (
                    sum(unimodal_terms) / max(1e-8, sum(weights))
                )
        if self.args.align_weight > 0 and "_aux_losses" in outputs:
            total = total + self.args.align_weight * sum(outputs["_aux_losses"].values())
        if self.args.moe_weight > 0 and "_moe_loss" in outputs:
            total = total + self.args.moe_weight * outputs["_moe_loss"]
        return total, mae.detach()

    @staticmethod
    def _accumulate_raw_moe(raw_sums, raw_counts, outputs, batch_size):
        raw = outputs.get("_moe_raw_losses", {}) or {}
        for key, value in raw.items():
            if not torch.is_tensor(value):
                continue
            raw_sums[key] = raw_sums.get(key, 0.0) + float(value.item()) * batch_size
            raw_counts[key] = raw_counts.get(key, 0) + batch_size

    @staticmethod
    def _moe_raw_average(raw_sums, raw_counts):
        return {
            key: round(raw_sums[key] / max(1, raw_counts.get(key, 1)), 6)
            for key in raw_sums
        }

    def train_epoch(self, model, loader, optimizer, scheduler, moe_aux_scale: float):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        grad_norm_sum = 0.0
        grad_norm_steps = 0
        skipped_batches = 0
        moe_raw_sums: dict = {}
        moe_raw_counts: dict = {}
        accum_steps = max(1, int(self.args.grad_accum_steps))
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        steps_in_accum = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc="train")):
            targets = batch["targets"].to(self.device).view(-1, 1)
            outputs = self.forward_batch(
                model,
                batch,
                labels=targets.view(-1),
                moe_aux_scale=moe_aux_scale,
            )
            loss, mae = self.compute_total_loss(outputs, targets)
            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                steps_in_accum = 0
                continue
            (loss / accum_steps).backward()
            steps_in_accum += 1
            ready_to_step = (steps_in_accum >= accum_steps) or (batch_idx == len(loader) - 1)
            if ready_to_step:
                grad_finite = True
                if self.args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.args.max_grad_norm,
                    )
                    grad_finite = torch.isfinite(grad_norm)
                    if grad_finite:
                        grad_norm_sum += grad_norm.item()
                        grad_norm_steps += 1
                if grad_finite:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                else:
                    skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                steps_in_accum = 0
            total_loss += loss.item() * targets.size(0)
            total_mae += mae.item() * targets.size(0)
            total_samples += targets.size(0)
            self._accumulate_raw_moe(moe_raw_sums, moe_raw_counts, outputs, targets.size(0))
        train_time_sec = time.perf_counter() - start
        return {
            "loss": round(total_loss / max(1, total_samples), 4),
            "mae": round(total_mae / max(1, total_samples), 4),
            "time_sec": train_time_sec,
            "throughput_samples_per_sec": total_samples / train_time_sec
            if train_time_sec > 0
            else 0.0,
            "samples": total_samples,
            "grad_norm": round(grad_norm_sum / max(1, grad_norm_steps), 4),
            "skipped_batches": skipped_batches,
            "lr": optimizer.param_groups[0]["lr"],
            "moe_raw_losses": self._moe_raw_average(moe_raw_sums, moe_raw_counts),
            "moe_aux_scale": moe_aux_scale,
        }

    @torch.no_grad()
    def evaluate(self, model, loader, split: str):
        model.eval()
        y_pred, y_true = [], []
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        start = time.perf_counter()
        for batch in tqdm(loader, desc=split):
            targets = batch["targets"].to(self.device).view(-1, 1)
            outputs = self.forward_batch(model, batch, labels=None, moe_aux_scale=0.0)
            loss, mae = self.compute_total_loss(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_mae += mae.item() * targets.size(0)
            total_samples += targets.size(0)
            y_pred.append(outputs["M"].detach().cpu())
            y_true.append(targets.detach().cpu())
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        abs_err = (pred.view(-1) - true.view(-1)).abs()
        abs_y = true.abs().view(-1)
        weak_mask = abs_y < self.args.moe_polarity_threshold
        strong_mask = ~weak_mask
        results["Loss_mae_weak"] = (
            round(abs_err[weak_mask].mean().item(), 4)
            if weak_mask.any()
            else float("nan")
        )
        results["Loss_mae_strong"] = (
            round(abs_err[strong_mask].mean().item(), 4)
            if strong_mask.any()
            else float("nan")
        )
        results["num_weak"] = int(weak_mask.sum().item())
        results["num_strong"] = int(strong_mask.sum().item())
        results["Loss"] = round(total_loss / max(1, total_samples), 4)
        results["Loss_mae"] = round(total_mae / max(1, total_samples), 4)
        results["eval_time_sec"] = time.perf_counter() - start
        printable = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        print(f"{split} >> " + " ".join(f"{k}: {v:.4f}" for k, v in printable.items()))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--piermoe_root", type=str, default="external/PIERMoE-main")
    parser.add_argument("--dataset", type=str, default="mosi", choices=["mosi", "mosei"])
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--desc_jsonl_path", type=str, default="")
    parser.add_argument("--model_save_path", "--output_dir", dest="model_save_path", type=str, default="checkpoints/piermoe")
    parser.add_argument("--log_dir", type=str, default="checkpoints/piermoe/logs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--align_weight", type=float, default=0.01)
    parser.add_argument("--unimodal_aux_weight", type=float, default=0.1)
    parser.add_argument("--unimodal_aux_text_weight", type=float, default=1.0)
    parser.add_argument("--unimodal_aux_audio_weight", type=float, default=2.0)
    parser.add_argument("--unimodal_aux_visual_weight", type=float, default=1.0)
    parser.add_argument("--return_auxiliary_heads", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--moe_weight", type=float, default=1.0)
    parser.add_argument("--moe_warmup_epochs", type=int, default=3)
    parser.add_argument("--text_model_name", type=str, default="models/text-backbone")
    parser.add_argument("--audio_model_name_or_path", type=str, default="models/audio-backbone")
    parser.add_argument("--desc_text_model_name", type=str, default="models/text-backbone")
    parser.add_argument("--hf_local_only", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--trust_remote_code", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument(
        "--backbone_torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32", "none"],
    )
    parser.add_argument("--visual_dim", type=int, default=768)
    parser.add_argument("--visual_num_frames", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attnres_num_blocks", type=int, default=4)
    parser.add_argument("--moe_polarity_guide_weight", type=float, default=0.3)
    parser.add_argument("--moe_interaction_pid_weight", type=float, default=0.1)
    parser.add_argument("--moe_router_entropy_weight", type=float, default=0.01)
    parser.add_argument("--moe_shared_alpha_init", type=float, default=0.2)
    parser.add_argument("--moe_polarity_threshold", type=float, default=-1.0)
    parser.add_argument("--moe_polarity_strong_threshold", type=float, default=-1.0)
    parser.add_argument("--desc_router_proj_dim", type=int, default=64)
    parser.add_argument("--desc_logit_scale_init", type=float, default=1.0)
    parser.add_argument("--use_triplet_uniqueness", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--persistent_workers", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--deterministic", default=True, type=lambda x: bool(strtobool(x)))
    args = parser.parse_args()

    args.moe_polarity_threshold, args.moe_polarity_strong_threshold = resolve_polarity_thresholds(args)
    sys.path.insert(0, args.piermoe_root)
    from safe_audio_data_loader import build_loaders_with_desc
    from utils.metricsTop import MetricsTop

    set_global_seed(args.seed, args.deterministic)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = dataset_root(args.dataset, args.data_root)
    train_loader, test_loader, val_loader = build_loaders_with_desc(
        root_dir=root_dir,
        batch_size=args.batch_size,
        desc_jsonl_path=args.desc_jsonl_path,
        use_visual=True,
        visual_num_frames=args.visual_num_frames,
        visual_dim=args.visual_dim,
        text_model_name=args.text_model_name,
        desc_text_model_name=args.desc_text_model_name,
        audio_feature_extractor_name=args.audio_model_name_or_path,
        hf_local_only=args.hf_local_only,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    args.return_auxiliary_heads = args.return_auxiliary_heads or args.unimodal_aux_weight > 0
    model_config = PIERMoEConfig(
        text_model_name=args.text_model_name,
        audio_model_name_or_path=args.audio_model_name_or_path,
        desc_text_model_name=args.desc_text_model_name,
        hf_local_only=args.hf_local_only,
        trust_remote_code=args.trust_remote_code,
        backbone_torch_dtype=args.backbone_torch_dtype,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        visual_dim=args.visual_dim,
        return_auxiliary_heads=args.return_auxiliary_heads,
        attnres_num_blocks=args.attnres_num_blocks,
        moe_polarity_guide_weight=args.moe_polarity_guide_weight,
        moe_interaction_pid_weight=args.moe_interaction_pid_weight,
        moe_router_entropy_weight=args.moe_router_entropy_weight,
        moe_shared_alpha_init=args.moe_shared_alpha_init,
        moe_polarity_threshold=args.moe_polarity_threshold,
        moe_polarity_strong_threshold=args.moe_polarity_strong_threshold,
        desc_router_proj_dim=args.desc_router_proj_dim,
        desc_logit_scale_init=args.desc_logit_scale_init,
        use_triplet_uniqueness=args.use_triplet_uniqueness,
        triplet_margin=args.triplet_margin,
    )
    model = PIERMoEForMER(model_config).to(device)
    model_stats = get_model_stats(model)
    print(f"PIERMoE config: {asdict(model_config)}")
    print(
        "MODEL >> total_params: {total_params}, trainable_params: {trainable_params}, "
        "frozen_params: {frozen_params}, model_size_mb: {model_size_mb:.2f}".format(
            **model_stats
        )
    )

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(optimizer=optimizer, args=args, total_steps=len(train_loader) * args.epochs)
    metrics = MetricsTop("regression").getMetics(args.dataset)
    trainer = Trainer(args, metrics, device)

    os.makedirs(args.model_save_path, exist_ok=True)
    lowest_val_mae = float("inf")
    highest_val_acc = -float("inf")
    best_mae_path = None
    best_mae_epoch = 0
    best_acc_epoch = 0
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_prefix = f"{args.dataset}_piermoe_seed{args.seed}_{run_stamp}"
    hyperparams = {"args": vars(args), "model_config": asdict(model_config)}
    logger = TrainLogger(log_dir=args.log_dir, run_name=ckpt_prefix, hyperparams=hyperparams)

    training_start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        moe_aux_scale = compute_aux_scale(epoch, args.moe_warmup_epochs)
        print(f"--------------------- EPOCH {epoch}  (moe_aux_scale={moe_aux_scale:.2f}) ---------------------")
        train_stats = trainer.train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            moe_aux_scale=moe_aux_scale,
        )
        moe_raw = train_stats.get("moe_raw_losses", {})
        moe_raw_str = " " + " ".join(f"{k}: {v}" for k, v in moe_raw.items()) if moe_raw else ""
        print(
            f"TRAIN >> total_loss: {train_stats['loss']:.4f} mae: {train_stats['mae']:.4f} "
            f"skipped: {train_stats['skipped_batches']}" + moe_raw_str
        )
        val_results = trainer.evaluate(model, val_loader, "VAL")
        epoch_time_sec = time.perf_counter() - epoch_start_time

        is_best_mae = False
        is_best_acc = False
        if val_results["Loss_mae"] < lowest_val_mae:
            if best_mae_path and os.path.exists(best_mae_path):
                os.remove(best_mae_path)
            lowest_val_mae = val_results["Loss_mae"]
            best_mae_epoch = epoch
            best_mae_path = os.path.join(
                args.model_save_path,
                f"best_mae_{ckpt_prefix}_{lowest_val_mae:.4f}.pth",
            )
            torch.save(
                {
                    "model_state_dict": trainable_state_dict(model),
                    "model_config": asdict(model_config),
                    "checkpoint_type": "trainable_only",
                },
                best_mae_path,
            )
            is_best_mae = True

        has0_acc = float(val_results.get("Has0_acc_2", 0.0))
        if has0_acc > highest_val_acc:
            highest_val_acc = has0_acc
            best_acc_epoch = epoch
            is_best_acc = True

        logger.log_epoch(
            {
                "epoch": epoch,
                "moe_aux_scale": moe_aux_scale,
                "train_loss": train_stats["loss"],
                "train_mae": train_stats["mae"],
                "train_time_sec": train_stats["time_sec"],
                "train_throughput_samples_per_sec": train_stats["throughput_samples_per_sec"],
                "train_samples": train_stats["samples"],
                "train_avg_grad_norm": train_stats["grad_norm"],
                "train_skipped_batches": train_stats["skipped_batches"],
                "train_moe_raw_losses": moe_raw,
                "lr": train_stats["lr"],
                "epoch_time_sec": epoch_time_sec,
                "val_results": val_results,
                "is_best_mae": is_best_mae,
                "is_best_acc": is_best_acc,
                "best_val_mae_so_far": lowest_val_mae,
                "best_val_acc_so_far": highest_val_acc,
                "best_mae_ckpt": best_mae_path,
                "best_acc_ckpt": "",
            }
        )

        last_improvement = max(best_mae_epoch, best_acc_epoch)
        if epoch >= args.moe_warmup_epochs and (epoch - last_improvement) >= args.early_stop:
            print(
                f"Early stop: no improvement for {args.early_stop} epochs "
                f"(last best at epoch {last_improvement})."
            )
            break

    total_training_time_sec = time.perf_counter() - training_start_time
    if best_mae_path is None:
        raise RuntimeError("No checkpoint was saved.")
    checkpoint = torch.load(best_mae_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    test_results = trainer.evaluate(model, test_loader, "TEST")
    print(f"Best mae checkpoint: {best_mae_path}")
    print(f"TEST lowest-val-mae Loss_mae: {test_results['Loss_mae']:.4f}")
    logger.log_summary(
        {
            "best_epoch_by_mae": best_mae_epoch,
            "best_epoch_by_acc": best_acc_epoch,
            "best_val_mae": lowest_val_mae,
            "best_val_acc": highest_val_acc,
            "training_time_sec": total_training_time_sec,
            "model_stats": model_stats,
            "best_mae_ckpt": best_mae_path,
            "best_acc_ckpt": "",
            "test_results_at_best_mae_ckpt": test_results,
        }
    )
    print(f"Training logs saved to: {logger.log_dir}")


if __name__ == "__main__":
    main()

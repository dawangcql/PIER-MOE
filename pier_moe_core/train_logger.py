from __future__ import annotations

import json
import os
from datetime import datetime


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


class TrainLogger:
    """MMML-style JSON logger for hyperparameters, epochs, and final summary."""

    def __init__(self, log_dir: str, run_name: str, hyperparams: dict):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.run_name = run_name
        self.hparam_path = os.path.join(self.log_dir, f"{self.run_name}_hparams.json")
        self.epoch_path = os.path.join(self.log_dir, f"{self.run_name}_epochs.jsonl")
        self.summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")

        payload = {
            "run_name": self.run_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": _to_serializable(hyperparams),
        }
        with open(self.hparam_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(self.epoch_path, "w", encoding="utf-8") as f:
            f.write("")

    def log_epoch(self, epoch_payload: dict) -> None:
        with open(self.epoch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_serializable(epoch_payload), ensure_ascii=False) + "\n")

    def log_summary(self, summary_payload: dict) -> None:
        payload = {
            "run_name": self.run_name,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": _to_serializable(summary_payload),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

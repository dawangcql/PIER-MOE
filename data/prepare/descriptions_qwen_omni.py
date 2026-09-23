"""Generate aligned MOSI/MOSEI descriptions using only Qwen2.5-Omni.

Audio and video phases share one checkpoint and write after every sample, so
jobs can be resumed safely with ``--resume``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_DEFAULT = "Qwen/Qwen2.5-Omni-7B"
VALIDATION_VERSION = 1

AUDIO_SYSTEM_PROMPT = (
    "You describe only directly audible acoustic properties. Never transcribe or summarize speech "
    "and never infer emotion, intent, topic, identity, or personality."
)
AUDIO_USER_PROMPT = (
    "Describe pitch and loudness variation, speaking rate, pauses, rhythm, voice quality, noise, "
    "and articulation clarity. Do not quote or paraphrase words. Return only JSON in the form "
    '{"status":"ok","description":"..."}; use status "too_short_or_unclear" and an empty '
    "description if the audio is insufficient."
)
AUDIO_RETRY_PROMPT = (
    "Your previous answer was invalid. Describe the audio again using only directly audible acoustic "
    "properties. Do not transcribe, summarize words, or mention emotion. Return only valid JSON with "
    "status and description."
)
VISUAL_SYSTEM_PROMPT = (
    "You describe only directly visible facial and head movements. Never infer emotion, identity, "
    "intent, personality, or scene meaning."
)
VISUAL_USER_PROMPT = (
    "Describe only visible movements of eyebrows, eyelids, gaze, lips, jaw, and head pose. Do not "
    "describe the scene, objects, clothing, or events. Return only JSON in the form "
    '{"status":"ok","description":"..."}; use status "no_face_or_too_short" and an empty '
    "description if the video is insufficient."
)
VISUAL_RETRY_PROMPT = (
    "Your previous answer was invalid. Describe the video again using only visible facial and head "
    "movements. Do not describe the scene, objects, or emotion. Return only valid JSON with status "
    "and description."
)

EMOTION_PATTERN = re.compile(
    r"\b(?:happy|sad|angry|nervous|calm|excited|afraid|surprised|emotion(?:al)?)\b|"
    r"开心|难过|愤怒|紧张",
    re.IGNORECASE,
)
AUDIO_FORBIDDEN = re.compile(
    r"\b(?:says|said|talks about|mentions|states|statement|transcript|"
    r"hello|thank you|i think|we are|you know)\b|[\"“”‘’]",
    re.IGNORECASE,
)
VISUAL_FORBIDDEN = re.compile(
    r"\b(?:room|background|screen|desk|chair|table|object|scene|"
    r"holding|walking|standing|sitting|clothing|wearing)\b",
    re.IGNORECASE,
)


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            records[str(item["sample_id"])] = item
        except (ValueError, KeyError):
            continue
    return records


def save_records(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records.values()),
        encoding="utf-8",
    )


def conversation(
    system_prompt: str,
    user_prompt: str,
    media_type: str,
    media_path: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": media_type, media_type: media_path},
            {"type": "text", "text": user_prompt},
        ]},
    ]


def generate(model: Any, processor: Any, messages: list[dict[str, Any]], max_tokens: int) -> str:
    from qwen_omni_utils import process_mm_info

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio=True)
    inputs = processor(
        text=prompt, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_tokens, return_audio=False)
    tokens = output[0] if isinstance(output, tuple) else output
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def normalize_description(description: str) -> str:
    return " ".join(description.split()).strip()


def valid_description(description: str, media_type: str) -> bool:
    description = normalize_description(description)
    forbidden = AUDIO_FORBIDDEN if media_type == "audio" else VISUAL_FORBIDDEN
    return bool(description) and not EMOTION_PATTERN.search(description) and not forbidden.search(description)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def json_candidates(text: str) -> list[dict[str, Any]]:
    text = strip_code_fence(text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return [payload]
    except ValueError:
        pass

    decoder = json.JSONDecoder()
    candidates = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
    return candidates


def parse_response(raw: str, media_type: str) -> tuple[str, str] | None:
    valid_insufficient_status = "too_short_or_unclear" if media_type == "audio" else "no_face_or_too_short"
    for payload in reversed(json_candidates(raw)):
        status = payload.get("status")
        description = payload.get("description")
        if not isinstance(status, str) or not isinstance(description, str):
            continue
        if status == "ok":
            description = normalize_description(description)
            if valid_description(description, media_type):
                return "ok", description
        elif status == valid_insufficient_status and not description.strip():
            return status, ""
    return None


def describe(
    model: Any,
    processor: Any,
    media_type: str,
    media_path: str,
    max_tokens: int,
) -> tuple[str, str, str, int]:
    if media_type == "audio":
        system_prompt = AUDIO_SYSTEM_PROMPT
        initial_prompt = AUDIO_USER_PROMPT
        retry_prompt = AUDIO_RETRY_PROMPT
    else:
        system_prompt = VISUAL_SYSTEM_PROMPT
        initial_prompt = VISUAL_USER_PROMPT
        retry_prompt = VISUAL_RETRY_PROMPT

    raw_output = ""
    for attempt, user_prompt in enumerate((initial_prompt, retry_prompt), start=1):
        raw_output = generate(
            model,
            processor,
            conversation(system_prompt, user_prompt, media_type, media_path),
            max_tokens,
        )
        parsed = parse_response(raw_output, media_type)
        if parsed is not None:
            status, description = parsed
            return status, description, raw_output, attempt
    return "invalid_output", "", raw_output, 2


def print_summary(
    records: dict[str, dict[str, Any]],
    sample_ids: set[str],
    fields: list[str],
) -> None:
    total_success = 0
    total_failure = 0
    print("\nDescription generation summary")
    for field in fields:
        statuses = [records.get(sample_id, {}).get(f"{field}_status", "not_run") for sample_id in sample_ids]
        success = sum(status == "ok" for status in statuses)
        failure = len(statuses) - success
        total_success += success
        total_failure += failure
        total = success + failure
        if total:
            print(
                f"{field}: total={total}, success={success} ({success / total:.2%}), "
                f"failure={failure} ({failure / total:.2%})"
            )
        else:
            print(f"{field}: total=0")

    total = total_success + total_failure
    if total:
        print(
            f"overall: total={total}, success={total_success} ({total_success / total:.2%}), "
            f"failure={total_failure} ({total_failure / total:.2%})"
        )
    else:
        print("overall: total=0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni descriptions for PIER-MoE")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("audio", "video", "all"), default="all")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = pd.read_csv(args.manifest).to_dict("records")
    fields = ["audio", "visual"] if args.phase == "all" else ["audio" if args.phase == "audio" else "visual"]

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    records = load_records(args.output) if args.resume else {}

    for row in rows:
        sample_id = str(row["sample_id"])
        record = records.setdefault(sample_id, {
            "sample_id": sample_id,
            "audio_status": "not_run",
            "visual_status": "not_run",
            "audio_description": "",
            "visual_description": "",
            "model": args.model,
        })
        jobs = []
        if args.phase in ("audio", "all"):
            jobs.append(("audio", "audio", str(row["wav_path"])))
        if args.phase in ("video", "all"):
            jobs.append(("visual", "video", str(row["mp4_path"])))

        for field, media_type, media_path in jobs:
            old_description = str(record.get(f"{field}_description", "") or "")
            if (
                args.resume
                and record.get(f"{field}_status") == "ok"
                and record.get(f"{field}_validation_version") == VALIDATION_VERSION
                and valid_description(old_description, media_type)
            ):
                continue

            record[f"{field}_description"] = ""
            record[f"{field}_attempts"] = 0
            record.pop(f"{field}_validation_version", None)
            record.pop(f"{field}_raw_output", None)
            try:
                status, description, raw_output, attempts = describe(
                    model, processor, media_type, media_path, args.max_new_tokens
                )
                record[f"{field}_status"] = status
                record[f"{field}_description"] = description
                record[f"{field}_raw_output"] = raw_output
                record[f"{field}_attempts"] = attempts
                if status == "ok":
                    record[f"{field}_validation_version"] = VALIDATION_VERSION
                record.pop(f"{field}_error", None)
            except Exception as exc:
                record[f"{field}_status"] = "error"
                record[f"{field}_error"] = str(exc)
            save_records(args.output, records)

    print_summary(records, {str(row["sample_id"]) for row in rows}, fields)


if __name__ == "__main__":
    main()

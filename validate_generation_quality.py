#!/usr/bin/env python3
"""Run a few held-out MEAD samples and sanity-check generation quality."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import wave
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CANONICAL_EMOTIONS = {
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AU generation quality on a few held-out MEAD test samples."
    )
    parser.add_argument(
        "--base-model-path",
        default="/home/user/D/Qwen-AU-finetune/Qwen-Audio-Chat",
        help="Path to the base Qwen-Audio-Chat model.",
    )
    parser.add_argument(
        "--lora-model-path",
        default="/home/user/D/checkpoints/qwen-audio-chat/checkpoint-2700",
        help="Path to the LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--test-inputs-jsonl",
        type=Path,
        default=Path("/home/user/D/mead_test_inputs.jsonl"),
        help="Path to the MEAD test inputs JSONL.",
    )
    parser.add_argument(
        "--test-refs-jsonl",
        type=Path,
        default=Path("/home/user/D/mead_test_refs.jsonl"),
        help="Path to the MEAD test refs JSONL.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of held-out samples to validate.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Device map to pass to transformers, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6144,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--print-full-response",
        action="store_true",
        help="Print the complete raw response for each validated sample.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def index_refs_by_id(path: Path) -> dict[str, dict]:
    return {record["id"]: record for record in load_jsonl(path)}


def resolve_torch_dtype(device_map: str):
    if device_map == "cpu":
        return torch.float32
    return torch.float16


def build_generation_config(model, max_new_tokens: int):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = max(
        getattr(generation_config, "max_length", 0) or 0,
        max_new_tokens + 8192,
    )
    generation_config.do_sample = False
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.temperature = 1.0
    return generation_config


def resolve_tokenizer_path(base_model_path: str, lora_model_path: str) -> str:
    lora_path = Path(lora_model_path)
    if (lora_path / "tokenizer_config.json").exists() and (
        lora_path / "tokenization_qwen.py"
    ).exists():
        return str(lora_path)
    return base_model_path


def patch_qwen_tokenizer_file(tokenizer_path: str) -> None:
    file_candidates = [
        Path(tokenizer_path) / "tokenization_qwen.py",
        Path.home()
        / ".cache/huggingface/modules/transformers_modules"
        / Path(tokenizer_path).name
        / "tokenization_qwen.py",
    ]
    old = """    def __init__(\n            self,\n            vocab_file,\n            errors=\"replace\",\n            audio_start_tag='<audio>',\n            audio_end_tag='</audio>',\n            **kwargs,\n    ):\n        super().__init__(**kwargs)\n        self.audio_start_tag = audio_start_tag\n"""
    new = """    def __init__(\n            self,\n            vocab_file,\n            errors=\"replace\",\n            audio_start_tag='<audio>',\n            audio_end_tag='</audio>',\n            **kwargs,\n    ):\n        self.audio_start_tag = audio_start_tag\n"""
    marker = """        self.im_end_id = self.special_tokens[IMEND]\n"""
    insertion = """        super().__init__(**kwargs)\n        self.im_end_id = self.special_tokens[IMEND]\n"""

    for file_path in file_candidates:
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8")
        if "self.AUDIO_ST" not in text or "super().__init__(**kwargs)" not in text:
            continue
        if old in text and insertion not in text:
            text = text.replace(old, new, 1)
            text = text.replace(marker, insertion, 1)
            file_path.write_text(text, encoding="utf-8")


def parse_response(response: str) -> tuple[str | None, list | None, str | None]:
    text = response.strip()
    if "," not in text:
        return None, None, "missing_comma_separator"

    emotion_label, au_part = text.split(",", 1)
    emotion_label = emotion_label.strip().lower()
    au_part = au_part.strip()

    if emotion_label not in CANONICAL_EMOTIONS:
        return emotion_label, None, "invalid_emotion_label"

    try:
        parsed = ast.literal_eval(au_part)
    except (SyntaxError, ValueError) as exc:
        return emotion_label, None, f"literal_eval_failed: {exc}"

    if not isinstance(parsed, list):
        return emotion_label, None, "parsed_au_sequence_is_not_a_list"

    return emotion_label, parsed, None


def load_audio_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_count = handle.getnframes()
        sample_rate = handle.getframerate()
    return frame_count / sample_rate


def first_frame_preview(sequence) -> str:
    if not sequence:
        return "[]"
    return repr(sequence[0])


def main() -> int:
    args = parse_args()

    refs_by_id = index_refs_by_id(args.test_refs_jsonl)
    test_records = load_jsonl(args.test_inputs_jsonl)[: args.num_samples]

    tokenizer_path = resolve_tokenizer_path(
        args.base_model_path, args.lora_model_path
    )
    patch_qwen_tokenizer_file(tokenizer_path)
    print(f"tokenizer_path={tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    tokenizer.model_max_length = max(tokenizer.model_max_length, 16384)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=args.device_map,
        torch_dtype=resolve_torch_dtype(args.device_map),
        trust_remote_code=True,
    ).eval()
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    model.generation_config.max_length = 16384
    generation_config = build_generation_config(model, args.max_new_tokens)

    for idx, record in enumerate(test_records, start=1):
        sample_id = record["id"]
        ref = refs_by_id[sample_id]
        user_message = record["messages"][0]
        audio_path = Path(user_message["audio"])

        query = tokenizer.from_list_format(
            [
                {"audio": user_message["audio"]},
                {"text": user_message["content"]},
            ]
        )
        response, _ = model.chat(
            tokenizer,
            query=query,
            history=None,
            generation_config=generation_config,
        )
        predicted_emotion, parsed_au, parse_error = parse_response(response)

        expected_au = ast.literal_eval(ref["au_sequence_string"])
        expected_frames = len(expected_au)
        audio_duration = load_audio_duration_seconds(audio_path)
        expected_frames_from_audio = int(round(audio_duration * 5))
        actual_frames = len(parsed_au) if parsed_au is not None else 0
        frame_match = (
            parsed_au is not None
            and abs(actual_frames - expected_frames_from_audio) <= 2
        )

        print(f"Sample {idx}: {sample_id}")
        print(
            f"  emotion: expected={ref['emotion_label']} predicted={predicted_emotion}"
        )
        print(
            "  parse_success: "
            f"{parse_error is None}"
            + ("" if parse_error is None else f" ({parse_error})")
        )
        print(
            "  frames: "
            f"ref_sequence={expected_frames} "
            f"audio_expected={expected_frames_from_audio} "
            f"predicted={actual_frames} "
            f"within_tolerance={frame_match}"
        )
        print(
            "  first_frame: "
            f"expected={first_frame_preview(expected_au)} "
            f"predicted={first_frame_preview(parsed_au or [])}"
        )
        print(f"  response_chars={len(response)}")
        if args.print_full_response:
            print("  raw_response_start")
            print(response)
            print("  raw_response_end")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

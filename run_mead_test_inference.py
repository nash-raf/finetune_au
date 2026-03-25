#!/usr/bin/env python3
"""Run Qwen-Audio inference on MEAD test inputs and save AU predictions."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference over MEAD test JSONL and save predicted AU JSON files."
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help="Path to the base Qwen-Audio-Chat model.",
    )
    parser.add_argument(
        "--lora-model-path",
        required=True,
        help="Path to the LoRA checkpoint or final adapter directory.",
    )
    parser.add_argument(
        "--test-inputs-jsonl",
        type=Path,
        required=True,
        help="Path to the MEAD test inputs JSONL.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory where predicted AU JSON files will be written.",
    )
    parser.add_argument(
        "--responses-jsonl",
        type=Path,
        required=True,
        help="Path to save raw model responses and parse status.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit on number of test samples to run. 0 means all.",
    )
    parser.add_argument(
        "--device-map",
        default="cuda",
        help="Device map to pass to transformers, for example 'cuda' or 'auto'.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate per sample.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling during generation. Defaults to greedy decoding.",
    )
    parser.add_argument(
        "--print-full-response",
        action="store_true",
        help="Print the complete raw response for each sample.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_generation_config(model, max_new_tokens: int, do_sample: bool):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = max(
        getattr(generation_config, "max_length", 0) or 0,
        max_new_tokens + 8192,
    )
    generation_config.do_sample = do_sample
    if not do_sample:
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


def ensure_base_model_support_files(base_model_path: str) -> None:
    base_path = Path(base_model_path)
    repo_root = Path(__file__).resolve().parent
    required = [
        "qwen_generation_utils.py",
        "configuration_qwen.py",
        "audio.py",
        "cpp_kernels.py",
    ]
    for name in required:
        dst = base_path / name
        src = repo_root / name
        if not dst.exists() and src.exists():
            shutil.copy(src, dst)


def resolve_torch_dtype(device_map: str):
    if device_map == "cpu":
        return torch.float32
    return torch.float16


def resolve_inner_model(model):
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    inner = getattr(model, "base_model", None)
    if inner is not None and hasattr(inner, "model"):
        return inner.model
    return model


def apply_qwen_length_overrides(model, max_new_tokens: int) -> None:
    target_max_length = max(max_new_tokens + 8192, 16384)
    models = [model]
    inner_model = resolve_inner_model(model)
    if inner_model is not model:
        models.append(inner_model)

    for item in models:
        if getattr(item, "generation_config", None) is not None:
            item.generation_config.max_length = max(
                getattr(item.generation_config, "max_length", 0) or 0,
                target_max_length,
            )
            item.generation_config.max_new_tokens = max_new_tokens
        config = getattr(item, "config", None)
        if config is not None:
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max(
                    getattr(config, "max_position_embeddings", 0) or 0,
                    target_max_length,
                )
            if hasattr(config, "seq_length"):
                config.seq_length = max(
                    getattr(config, "seq_length", 0) or 0,
                    target_max_length,
                )


def parse_response(response: str) -> tuple[str | None, list | None, str | None]:
    text = response.strip()
    if "," not in text:
        return None, None, "missing_comma_separator"

    emotion_label, au_part = text.split(",", 1)
    emotion_label = emotion_label.strip().lower()
    au_part = au_part.strip()

    try:
        parsed = ast.literal_eval(au_part)
    except (SyntaxError, ValueError) as exc:
        return emotion_label, None, f"literal_eval_failed: {exc}"

    if not isinstance(parsed, list):
        return emotion_label, None, "parsed_au_sequence_is_not_a_list"

    return emotion_label, parsed, None


def main() -> int:
    args = parse_args()

    tokenizer_path = resolve_tokenizer_path(
        args.base_model_path, args.lora_model_path
    )
    ensure_base_model_support_files(args.base_model_path)
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
    apply_qwen_length_overrides(model, args.max_new_tokens)
    generation_config = build_generation_config(
        model, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample
    )

    records = load_records(args.test_inputs_jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.responses_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.responses_jsonl.open("w", encoding="utf-8") as responses_handle:
        for idx, record in enumerate(records, start=1):
            sample_id = record["id"]
            messages = record["messages"]
            user_message = messages[0]

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
            emotion_label, au_sequence, error = parse_response(response)

            pred_json_path = args.pred_dir / f"{sample_id}.json"
            pred_json_path.parent.mkdir(parents=True, exist_ok=True)

            if au_sequence is not None:
                with pred_json_path.open("w", encoding="utf-8") as pred_handle:
                    json.dump(au_sequence, pred_handle)

            result = {
                "id": sample_id,
                "audio": user_message["audio"],
                "raw_response": response,
                "predicted_emotion": emotion_label,
                "pred_json_path": str(pred_json_path),
                "parse_error": error,
                "au_json_written": au_sequence is not None,
            }
            responses_handle.write(json.dumps(result, ensure_ascii=False) + "\n")

            status = "ok" if error is None else f"parse_error={error}"
            print(
                f"[{idx}/{len(records)}] {sample_id} -> {status} "
                f"(chars={len(response)})"
            )
            if args.print_full_response:
                print("RAW_RESPONSE_START")
                print(response)
                print("RAW_RESPONSE_END")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

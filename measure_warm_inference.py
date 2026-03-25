#!/usr/bin/env python3
"""Measure cold-start and warm inference time for one MEAD sample."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure cold and warm inference time for one test sample."
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
        "--sample-index",
        type=int,
        default=0,
        help="0-based sample index inside the test JSONL.",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=2,
        help="Number of timed warm runs after model loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device-map",
        default="cuda",
        help="Device map to pass to transformers.",
    )
    parser.add_argument(
        "--print-full-response",
        action="store_true",
        help="Print the complete raw response for each timed run.",
    )
    parser.add_argument(
        "--debug-breakdown",
        action="store_true",
        help="Print fine-grained timing for audio preprocessing, context building, generation, and forward passes.",
    )
    return parser.parse_args()


def load_record(path: Path, index: int) -> dict:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if index < 0 or index >= len(records):
        raise IndexError(f"sample index {index} out of range for {len(records)} records")
    return records[index]


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


def sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def instrument_inference(model, tokenizer, enabled: bool):
    if not enabled:
        return None, lambda: None

    timings = {
        "process_audio_seconds": 0.0,
        "make_context_seconds": 0.0,
        "audio_encode_seconds": 0.0,
        "generate_seconds": 0.0,
        "forward_seconds": 0.0,
    }
    counts = {
        "process_audio_calls": 0,
        "make_context_calls": 0,
        "audio_encode_calls": 0,
        "generate_calls": 0,
        "forward_calls": 0,
    }

    model_module = sys.modules[model.__class__.__module__]
    orig_make_context = getattr(model_module, "make_context", None)
    orig_process_audio = getattr(tokenizer, "process_audio", None)
    orig_audio_encode = getattr(model.transformer.audio, "encode", None)
    orig_generate = model.generate
    orig_forward = model.forward

    def timed_call(key_seconds, key_calls, fn, *args, **kwargs):
        sync_if_needed()
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        sync_if_needed()
        timings[key_seconds] += time.perf_counter() - started
        counts[key_calls] += 1
        return result

    if orig_make_context is not None:
        @functools.wraps(orig_make_context)
        def wrapped_make_context(*args, **kwargs):
            return timed_call(
                "make_context_seconds",
                "make_context_calls",
                orig_make_context,
                *args,
                **kwargs,
            )

        setattr(model_module, "make_context", wrapped_make_context)

    if orig_process_audio is not None:
        @functools.wraps(orig_process_audio)
        def wrapped_process_audio(*args, **kwargs):
            return timed_call(
                "process_audio_seconds",
                "process_audio_calls",
                orig_process_audio,
                *args,
                **kwargs,
            )

        tokenizer.process_audio = wrapped_process_audio

    if orig_audio_encode is not None:
        @functools.wraps(orig_audio_encode)
        def wrapped_audio_encode(*args, **kwargs):
            return timed_call(
                "audio_encode_seconds",
                "audio_encode_calls",
                orig_audio_encode,
                *args,
                **kwargs,
            )

        model.transformer.audio.encode = wrapped_audio_encode

    @functools.wraps(orig_generate)
    def wrapped_generate(*args, **kwargs):
        return timed_call(
            "generate_seconds",
            "generate_calls",
            orig_generate,
            *args,
            **kwargs,
        )

    @functools.wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        return timed_call(
            "forward_seconds",
            "forward_calls",
            orig_forward,
            *args,
            **kwargs,
        )

    model.generate = wrapped_generate
    model.forward = wrapped_forward

    def reset_timings():
        for key in timings:
            timings[key] = 0.0
        for key in counts:
            counts[key] = 0

    return (timings, counts), reset_timings


def main() -> int:
    args = parse_args()

    load_started = time.perf_counter()
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
    model.generation_config.max_length = 16384
    generation_config = build_generation_config(model, args.max_new_tokens)
    load_elapsed = time.perf_counter() - load_started
    instrumentation, reset_timings = instrument_inference(
        model, tokenizer, enabled=args.debug_breakdown
    )

    record = load_record(args.test_inputs_jsonl, args.sample_index)
    user_message = record["messages"][0]
    query = tokenizer.from_list_format(
        [
            {"audio": user_message["audio"]},
            {"text": user_message["content"]},
        ]
    )

    print(f"sample_id={record['id']}")
    print(f"model_load_seconds={load_elapsed:.3f}")

    for run_idx in range(1, args.warm_runs + 1):
        reset_timings()
        started = time.perf_counter()
        response, _ = model.chat(
            tokenizer,
            query=query,
            history=None,
            generation_config=generation_config,
        )
        elapsed = time.perf_counter() - started
        print(f"run_{run_idx}_seconds={elapsed:.3f}")
        print(f"run_{run_idx}_response_chars={len(response)}")
        print(f"run_{run_idx}_response_prefix={response[:200]!r}")
        if instrumentation is not None:
            timings, counts = instrumentation
            print(
                f"run_{run_idx}_process_audio_seconds={timings['process_audio_seconds']:.3f}"
            )
            print(
                f"run_{run_idx}_process_audio_calls={counts['process_audio_calls']}"
            )
            print(
                f"run_{run_idx}_make_context_seconds={timings['make_context_seconds']:.3f}"
            )
            print(
                f"run_{run_idx}_make_context_calls={counts['make_context_calls']}"
            )
            print(
                f"run_{run_idx}_audio_encode_seconds={timings['audio_encode_seconds']:.3f}"
            )
            print(
                f"run_{run_idx}_audio_encode_calls={counts['audio_encode_calls']}"
            )
            print(
                f"run_{run_idx}_generate_seconds={timings['generate_seconds']:.3f}"
            )
            print(f"run_{run_idx}_generate_calls={counts['generate_calls']}")
            print(
                f"run_{run_idx}_forward_seconds={timings['forward_seconds']:.3f}"
            )
            print(f"run_{run_idx}_forward_calls={counts['forward_calls']}")
            if counts["forward_calls"] > 0:
                print(
                    "run_{}_avg_forward_seconds={:.6f}".format(
                        run_idx,
                        timings["forward_seconds"] / counts["forward_calls"],
                    )
                )
        if args.print_full_response:
            print(f"run_{run_idx}_response_full_start")
            print(response)
            print(f"run_{run_idx}_response_full_end")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

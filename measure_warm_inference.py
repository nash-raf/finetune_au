#!/usr/bin/env python3
"""Measure cold-start and warm inference time for one MEAD sample."""

from __future__ import annotations

import argparse
import copy
import json
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


def main() -> int:
    args = parse_args()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )
    tokenizer.model_max_length = max(tokenizer.model_max_length, 16384)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    generation_config = build_generation_config(model, args.max_new_tokens)
    load_elapsed = time.perf_counter() - load_started

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

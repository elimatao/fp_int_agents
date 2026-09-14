#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["mlx-lm"]
# ///
"""Compare base Qwen2.5-3B vs. LoRA fine-tuned model on conversation test data."""

import argparse
import json
import os
import tomllib
import types
from pathlib import Path

_config_file = Path(__file__).parent / "config.toml"
if _config_file.exists():
    _config = tomllib.loads(_config_file.read_text())
    if "HF_KEY" in _config:
        os.environ.setdefault("HF_TOKEN", _config["HF_KEY"])

from mlx_lm import generate, load
from mlx_lm.lora import CONFIG_DEFAULTS, run

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-bf16"
DEFAULT_ADAPTER = Path("adapters_lr_e_-4")


def load_test_samples(path: Path, max_samples: int) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )
            asst_msg = next(
                (m["content"] for m in messages if m["role"] == "assistant"), None
            )
            if user_msg and asst_msg:
                samples.append({"question": user_msg, "answer": asst_msg})
            if len(samples) >= max_samples:
                break
    return samples


def compare_generations(
    model_name: str,
    adapter_path: Path,
    samples: list[dict],
    max_tokens: int,
) -> None:
    print("\n" + "=" * 70)
    print("1. LOADING BASE MODEL FOR QUALITATIVE INFERENCE...")
    print("=" * 70)
    base_model, tokenizer = load(model_name)

    base_responses = []
    for i, s in enumerate(samples):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": s["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"Generating base response {i + 1}/{len(samples)}...")
        resp = generate(
            base_model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        base_responses.append(resp.strip())

    # Free base model weights before loading fine-tuned model
    del base_model

    print("\n" + "=" * 70)
    print("2. LOADING FINE-TUNED MODEL (WITH LORA ADAPTER)...")
    print("=" * 70)
    ft_model, ft_tokenizer = load(model_name, adapter_path=str(adapter_path))

    ft_responses = []
    for i, s in enumerate(samples):
        prompt = ft_tokenizer.apply_chat_template(
            [{"role": "user", "content": s["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"Generating fine-tuned response {i + 1}/{len(samples)}...")
        resp = generate(
            ft_model,
            ft_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        ft_responses.append(resp.strip())

    del ft_model

    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON RESULTS")
    print("=" * 70)

    for i, (sample, base_r, ft_r) in enumerate(
        zip(samples, base_responses, ft_responses, strict=False)
    ):
        print(f"\n--- SAMPLE {i + 1} ---")
        print(f"[QUESTION]:\n{sample['question']}\n")
        print(f"[GROUND TRUTH ANSWER (Bundestag)]:\n{sample['answer'][:400]}...\n")
        print(f"[BASE MODEL ANSWER]:\n{base_r}\n")
        print(f"[FINE-TUNED MODEL ANSWER]:\n{ft_r}\n")
        print("-" * 70)


def evaluate_test_perplexity(
    model_name: str,
    data_dir: Path,
    adapter_path: Path,
    test_batches: int,
) -> None:
    print("\n" + "=" * 70)
    print("3. QUANTITATIVE EVALUATION (TEST SET PERPLEXITY)")
    print("=" * 70)

    print("\n--> Evaluating Base Model on test set:")
    cfg_base = {
        **CONFIG_DEFAULTS,
        "model": model_name,
        "data": str(data_dir),
        "train": False,
        "test": True,
        "adapter_path": "",
        "test_batches": test_batches,
        "max_seq_length": 2048,
    }
    run(types.SimpleNamespace(**cfg_base))

    print(f"\n--> Evaluating Fine-Tuned Model ({adapter_path}) on test set:")
    cfg_ft = {
        **CONFIG_DEFAULTS,
        "model": model_name,
        "data": str(data_dir),
        "train": False,
        "test": True,
        "adapter_path": str(adapter_path),
        "test_batches": test_batches,
        "max_seq_length": 2048,
    }
    run(types.SimpleNamespace(**cfg_ft))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Base vs. Fine-Tuned Conversation Model"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("scripts/kleine_anfragen/conversation_wp20"),
        help="Directory with test.jsonl",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model path or HF repo (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER,
        help=f"Path to trained adapter (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of qualitative test samples to generate (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per response (default: 256)",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=100,
        help="Number of test batches to evaluate for perplexity (default: 100)",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip test perplexity computation and only do qualitative generation",
    )
    args = parser.parse_args()

    # Fallback to 'adapters' if DEFAULT_ADAPTER does not exist
    adapter_dir = args.adapter_path
    if not adapter_dir.exists():
        fallback = Path("adapters")
        if fallback.exists():
            adapter_dir = fallback
        else:
            print(f"Warning: adapter directory {args.adapter_path} not found.")

    test_path = args.data / "test.jsonl"
    if not test_path.exists():
        print(f"Error: test file not found at {test_path}")
        return

    samples = load_test_samples(test_path, args.num_samples)
    print(f"Loaded {len(samples)} qualitative samples from {test_path}")

    compare_generations(
        model_name=str(args.model),
        adapter_path=adapter_dir,
        samples=samples,
        max_tokens=args.max_tokens,
    )

    if not args.skip_perplexity:
        evaluate_test_perplexity(
            model_name=str(args.model),
            data_dir=args.data,
            adapter_path=adapter_dir,
            test_batches=args.test_batches,
        )


if __name__ == "__main__":
    main()

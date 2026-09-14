#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["sentence-transformers", "peft", "datasets"]
# ///
"""Evaluate and compare base cross-encoder vs. fine-tuned LoRA reranker on test.jsonl."""

import argparse
import json
import os
import tomllib
from pathlib import Path

# Prevent PyTorch MPS from hoarding cached Metal memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
)

_config_file = Path(__file__).parent / "config.toml"
if _config_file.exists():
    _config = tomllib.loads(_config_file.read_text())
    if "HF_KEY" in _config:
        os.environ.setdefault("HF_TOKEN", _config["HF_KEY"])

DEFAULT_BASE = "BAAI/bge-reranker-v2-m3"
DEFAULT_ADAPTER = Path("adapters_reranker/final")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_model(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
    labels: list[int],
    name: str,
    batch_size: int,
) -> dict[str, float]:
    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=pairs,
        labels=labels,
        name=name,
        batch_size=batch_size,
        write_csv=False,
    )
    return evaluator(model)


def print_comparison_table(
    base_results: dict[str, float],
    ft_results: dict[str, float],
    metrics: list[tuple[str, str]],
) -> None:
    print("\n" + "=" * 62)
    print(f"{'Metric':<22} | {'Base Model':<12} | {'Fine-Tuned':<12} | {'Delta':<8}")
    print("-" * 62)
    for display_name, key in metrics:
        b_val = base_results.get(f"base_{key}", 0.0)
        ft_val = ft_results.get(f"ft_{key}", 0.0)
        delta = ft_val - b_val
        delta_str = f"{delta:+.4f}" if delta != 0 else " 0.0000"
        print(f"{display_name:<22} | {b_val:<12.4f} | {ft_val:<12.4f} | {delta_str:<8}")
    print("=" * 62 + "\n")


def print_qualitative_samples(
    base_model: CrossEncoder,
    ft_model: CrossEncoder,
    records: list[dict],
    num_samples: int = 2,
) -> None:
    print("=" * 62)
    print("QUALITATIVE COMPARISON SAMPLES")
    print("=" * 62)

    sample_records = records[: num_samples * 2]
    pairs = [(r["query"], r["passage"]) for r in sample_records]
    labels = [int(r["label"]) for r in sample_records]

    base_scores = base_model.predict(pairs)
    ft_scores = ft_model.predict(pairs)

    for i, (pair, label, b_s, ft_s) in enumerate(
        zip(pairs, labels, base_scores, ft_scores, strict=False)
    ):
        label_text = (
            "POSITIVE (Ground Truth)" if label == 1 else "NEGATIVE (Distractor)"
        )
        print(f"\nSample {i + 1} [{label_text}]:")
        print(f"  Query:   {pair[0][:120]}...")
        print(f"  Passage: {pair[1][:120]}...")
        print(f"  Base Score:       {float(b_s):.4f}")
        print(f"  Fine-Tuned Score: {float(ft_s):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Base vs. Fine-Tuned Cross-Encoder on Test Set"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("scripts/kleine_anfragen/reranker_wp20/test.jsonl"),
        help="Path to test.jsonl",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help=f"Base model ID or path (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_ADAPTER,
        help=f"Directory of fine-tuned adapter (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length (default: 512)",
    )
    args = parser.parse_args()

    if not args.test_file.exists():
        print(f"Error: test file not found at {args.test_file}")
        return

    records = load_jsonl(args.test_file)
    print(f"Loaded {len(records)} test examples from {args.test_file}")

    pairs = [(r["query"], r["passage"]) for r in records]
    labels = [int(r["label"]) for r in records]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\n1. Evaluating Base Model: {args.base_model} ...")
    base_model = CrossEncoder(
        str(args.base_model),
        num_labels=1,
        max_length=args.max_length,
        device=device,
    )
    base_results = evaluate_model(
        base_model, pairs, labels, name="base", batch_size=args.batch_size
    )

    print(f"\n2. Evaluating Fine-Tuned Model: {args.adapter_dir} ...")
    ft_model = CrossEncoder(
        str(args.adapter_dir),
        num_labels=1,
        max_length=args.max_length,
        device=device,
    )
    ft_results = evaluate_model(
        ft_model, pairs, labels, name="ft", batch_size=args.batch_size
    )

    metrics = [
        ("Accuracy", "accuracy"),
        ("Average Precision", "average_precision"),
        ("F1 Score", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
    ]

    print_comparison_table(base_results, ft_results, metrics)
    print_qualitative_samples(base_model, ft_model, records, num_samples=2)


if __name__ == "__main__":
    main()

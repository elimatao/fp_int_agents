#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["sentence-transformers", "peft", "datasets"]
# ///
"""Fine-tune a cross-encoder reranker with LoRA via sentence-transformers + PEFT.

Runs on Apple Silicon via PyTorch MPS.

Expects a directory produced by generate_reranker_dataset.py --split,
which contains train.jsonl, val.jsonl, and test.jsonl.

Usage:
    python finetune_reranker.py --data data/ --output models/reranker/
    python finetune_reranker.py --data data/ --output models/reranker/ --epochs 3 --rank 8
"""

import argparse
import json
import math
import os
import tomllib
from pathlib import Path

# Prevent PyTorch MPS from hoarding cached Metal memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

_config_file = Path(__file__).parent / "config.toml"
if _config_file.exists():
    _config = tomllib.loads(_config_file.read_text())
    if "HF_KEY" in _config:
        os.environ.setdefault("HF_TOKEN", _config["HF_KEY"])

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class ClearMPSCallback(TrainerCallback):
    """Periodically flush cached Metal allocations to prevent MPS memory leaks."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if state.global_step % 10 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "sentence_A": [r["query"] for r in records],
            "sentence_B": [r["passage"] for r in records],
            "label": [float(r["label"]) for r in records],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune cross-encoder reranker"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory with train.jsonl and val.jsonl (output of generate_reranker_dataset.py --split)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model path or HF repo (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/reranker"),
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Per-device batch size (default: 8)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2 -> effective batch size 16)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512, covers ~94%% of pairs while preventing OOM)",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA (default: 1e-4)",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_records = load_jsonl(args.data / "train.jsonl")
    val_records = load_jsonl(args.data / "val.jsonl")

    n_train = len(train_records)
    effective_batch = args.batch_size * args.grad_accum
    total_steps = math.ceil(n_train / effective_batch) * args.epochs

    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    print(
        f"Data:       {args.data}  ({n_train} train / {len(val_records)} val examples)"
    )
    print(f"Output:     {args.output}")
    print(f"Max length: {args.max_length}")
    print(
        f"Epochs:     {args.epochs}  →  {total_steps} steps  |  rank={args.rank}  |  "
        f"batch={args.batch_size}x{args.grad_accum}  |  lr={args.learning_rate}"
    )
    print()

    model = CrossEncoder(
        str(args.model),
        num_labels=1,
        max_length=args.max_length,
        device=device,
    )
    model.add_adapter(
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            target_modules="all-linear",
        )
    )

    loss = BinaryCrossEntropyLoss(model)

    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=[(r["query"], r["passage"]) for r in val_records],
        labels=[int(r["label"]) for r in val_records],
        name="val",
        batch_size=args.batch_size,
    )

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        bf16=False,
        fp16=False,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=0.02,
        dataloader_pin_memory=False,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=to_dataset(train_records),
        eval_dataset=to_dataset(val_records),
        loss=loss,
        evaluator=evaluator,
        callbacks=[ClearMPSCallback()],
    )
    trainer.train()

    final_dir = args.output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(str(final_dir))
    print(f"\nModel and adapter saved to {final_dir}")

    print("\nSanity check predictions on validation samples:")
    sample_pairs = [(r["query"], r["passage"]) for r in val_records[:2]]
    sample_labels = [r["label"] for r in val_records[:2]]
    scores = model.predict(sample_pairs)
    for pair, label, score in zip(sample_pairs, sample_labels, scores, strict=False):
        print(
            f"  Label: {label} | Predicted Score: {float(score):.4f} | Query: {pair[0][:80]}..."
        )


if __name__ == "__main__":
    main()

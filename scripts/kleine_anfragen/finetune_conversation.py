#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["mlx-lm"]
# ///
"""Fine-tune Qwen2.5-3B on conversation data using LoRA via MLX-LM.

Expects a directory produced by generate_conversation_dataset.py --split,
which contains train.jsonl, valid.jsonl, and test.jsonl.

Usage:
    python finetune_conversation.py --data data/ --output adapters/ --epochs 3
    python finetune_conversation.py --data data/ --output adapters/ --epochs 3 --rank 8
"""

import argparse
import math
import os
import tomllib
import types
from pathlib import Path

_config_file = Path(__file__).parent / "config.toml"
if _config_file.exists():
    _config = tomllib.loads(_config_file.read_text())
    if "HF_KEY" in _config:
        os.environ.setdefault("HF_TOKEN", _config["HF_KEY"])

from mlx_lm.lora import CONFIG_DEFAULTS, run

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-bf16"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-3B with MLX-LM"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory with train.jsonl and valid.jsonl (output of generate_conversation_dataset.py --split)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model path or HF repo (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adapters"),
        help="Adapter output directory (default: adapters)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=16,
        help="Number of layers to adapt (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "adam", "adafactor", "sgd"],
        help="Optimizer (default: adamw)",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=300,
        help="Number of steps between evaluations (default: 300)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=300,
        help="Save checkpoint every N steps (default: 300)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="ADAPTER_FILE",
        help="Resume training from a checkpoint, e.g. adapters/0000300_adapters.safetensors",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: random)"
    )
    args = parser.parse_args()

    n_train = sum(1 for _ in (args.data / "train.jsonl").open())
    iters = math.ceil(n_train / args.batch_size) * args.epochs

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Model:           {args.model}")
    print(f"Data:            {args.data}  ({n_train} train examples)")
    print(f"Adapters:        {args.output}")
    print(
        f"Epochs:          {args.epochs}  →  {iters} iters  |  rank={args.rank}  |  "
        f"layers={args.num_layers}  |  batch={args.batch_size}  |  lr={args.learning_rate}  |  "
        f"opt={args.optimizer}"
    )
    print(f"Grad checkpoint: True  |  eval/save every={args.save_every} steps")
    print()

    cfg = {
        **CONFIG_DEFAULTS,
        "model": str(args.model),
        "train": True,
        "data": str(args.data),
        "batch_size": args.batch_size,
        "iters": iters,
        "learning_rate": args.learning_rate,
        "adapter_path": str(args.output),
        "lora_parameters": {**CONFIG_DEFAULTS["lora_parameters"], "rank": args.rank},
        "max_seq_length": 2048,
        "optimizer": args.optimizer,
        "grad_checkpoint": True,
        "num_layers": args.num_layers,
        "steps_per_eval": args.steps_per_eval,
        "save_every": args.save_every,
        "resume_adapter_file": str(args.resume) if args.resume else None,
        "mask_prompt": True,
        **({"seed": args.seed} if args.seed is not None else {}),
    }
    run(types.SimpleNamespace(**cfg))


if __name__ == "__main__":
    main()

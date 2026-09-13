"""
Generate a conversational fine-tuning dataset from kleine_anfragen QA pairs.

Output format: ChatML-compatible JSONL with single-turn Q/A exchanges.
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Compatible with MLX-LM, LLaMA-Factory, Axolotl, and Unsloth.
Supports train/val/test splitting via --split (same document-level split as the reranker script).
"""

import argparse
import json
import sys
from pathlib import Path

from generate_reranker_dataset import clean_question, load_documents, split_documents


def generate_examples(docs: list[dict]) -> list[dict]:
    examples = []
    for doc in docs:
        for pair in doc.get("pairs", []):
            question = clean_question(pair["question"])
            answer = pair["answer"]
            if question and answer:
                examples.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                })
    return examples


def write_examples(examples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conversational fine-tuning dataset")
    parser.add_argument("input", type=Path, help="Input JSONL file with QA pairs")
    parser.add_argument("output", type=Path, help="Output JSONL file (with --split: output directory)")
    parser.add_argument("--split", action="store_true", help="Split into train/valid/test (output must be a directory)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of documents for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    docs = load_documents(args.input)
    print(f"Loaded {len(docs)} documents", file=sys.stderr)

    if not args.split:
        examples = generate_examples(docs)
        print(f"Generated {len(examples)} examples", file=sys.stderr)
        write_examples(examples, args.output)
        print(f"Written to {args.output}", file=sys.stderr)
        return

    if not (0.0 < args.train_ratio < 1.0):
        print("Error: --train-ratio must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    train_docs, val_docs, test_docs = split_documents(docs, args.train_ratio, args.seed)
    print(
        f"Split: {len(train_docs)} train / {len(val_docs)} val / {len(test_docs)} test documents",
        file=sys.stderr,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    # MLX-LM expects valid.jsonl (not val.jsonl)
    split_names = [("train", train_docs), ("valid", val_docs), ("test", test_docs)]
    for name, split_docs in split_names:
        examples = generate_examples(split_docs)
        out_path = args.output / f"{name}.jsonl"
        print(f"{name}: {len(examples)} examples", file=sys.stderr)
        write_examples(examples, out_path)
        print(f"Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

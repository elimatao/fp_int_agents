"""
Generate a cross-encoder reranker training dataset from kleine_anfragen QA pairs.

Positives: each (question, answer) pair.
Negatives: same document, answer shifted by 1 in round-robin (skips single-pair docs).

Output format: JSONL with {"query": str, "passage": str, "label": int}

With --split: outputs train/val/test splits instead of a single file.
Documents are split by vorgang_id to avoid leakage across splits.
Use --train-ratio to control split sizes (val and test share the remainder equally).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Matches leading question numbers like "1.", "12.", "1. 2." (combined questions)
_LEADING_NUMBER_RE = re.compile(r"^(\d+\.\s*)+")


def clean_question(text: str) -> str:
    return _LEADING_NUMBER_RE.sub("", text).strip()


def load_documents(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_no}: {e}", file=sys.stderr)
    return docs


def generate_examples(docs: list[dict]) -> list[dict]:
    examples = []

    for doc in docs:
        pairs = doc.get("pairs", [])
        n = len(pairs)

        if n == 0:
            continue

        for i, pair in enumerate(pairs):
            question = pair["question"]
            answer = pair["answer"]

            query = clean_question(question)
            examples.append({"query": query, "passage": answer, "label": 1})

            if n > 1:
                negative_answer = pairs[(i + 1) % n]["answer"]
                examples.append(
                    {"query": query, "passage": negative_answer, "label": 0}
                )

    return examples


def split_documents(
    docs: list[dict], train_ratio: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = docs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_remaining = n - n_train
    n_val = n_remaining // 2

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_examples(examples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def report(name: str, examples: list[dict]) -> None:
    pos = sum(1 for e in examples if e["label"] == 1)
    neg = sum(1 for e in examples if e["label"] == 0)
    print(
        f"{name}: {len(examples)} examples ({pos} positive, {neg} negative)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reranker training dataset")
    parser.add_argument("input", type=Path, help="Input JSONL file with QA pairs")
    parser.add_argument(
        "output", type=Path, help="Output JSONL file (with --split: output directory)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test (output must be a directory)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of documents for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    args = parser.parse_args()

    docs = load_documents(args.input)
    print(f"Loaded {len(docs)} documents", file=sys.stderr)

    if not args.split:
        examples = generate_examples(docs)
        report("output", examples)
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
    for name, split_docs in [
        ("train", train_docs),
        ("val", val_docs),
        ("test", test_docs),
    ]:
        out_path = args.output / f"{name}.jsonl"
        examples = generate_examples(split_docs)
        report(name, examples)
        write_examples(examples, out_path)
        print(f"Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

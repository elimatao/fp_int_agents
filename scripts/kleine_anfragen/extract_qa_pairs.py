#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pdfplumber"]
# ///
"""Extract Q/A pairs from Antwort PDFs listed in a fetch index.

Output: one JSONL line per PDF with all Q/A pairs in document order.
PDFs without a text layer are skipped and reported.

Output format per line:
  {"pdf_file": "20-15149.pdf", "vorgang_id": "...", "pairs": [{"question": "...", "answer": "..."}, ...]}
"""

import argparse
import json
import re
from pathlib import Path

import pdfplumber

# Font size thresholds — question text is set ~1pt smaller than answer text.
# Observed: WP17/19/20 use 9.6/10.7, WP18 uses 9.5/10.5.
# We classify a line as a question if its dominant font size is below this threshold.
QUESTION_SIZE_MAX = 9.8
MIN_CHARS_FOR_TEXT_LAYER = 100


def extract_lines(pdf_path: Path) -> list[tuple[float, str]] | None:
    """Return list of (font_size, text) lines, or None if no text layer."""
    with pdfplumber.open(pdf_path) as pdf:
        total_chars = sum(len(p.chars) for p in pdf.pages)
        if total_chars < MIN_CHARS_FOR_TEXT_LAYER:
            return None

        lines = []
        for page_idx, page in enumerate(pdf.pages):
            # Headers on continuation pages sit at y≈62–63; real content starts at y≥73.
            header_cutoff = 70 if page_idx > 0 else 0
            words = page.extract_words(extra_attrs=["size"])
            by_y: dict[int, list] = {}
            for w in words:
                y = round(w["top"])
                if y < header_cutoff:
                    continue
                by_y.setdefault(y, []).append(w)
            for y in sorted(by_y):
                row = sorted(by_y[y], key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in row).strip()
                size = round(row[0]["size"], 1)
                if text:
                    lines.append((size, text))

    return lines


_SOFT_HYPHEN_RE = re.compile(r"-\s+(?=[a-zäöüß])")


def join_lines(lines: list[str]) -> str:
    """Join lines, merging words broken across lines with a soft hyphen."""
    return _SOFT_HYPHEN_RE.sub("", " ".join(lines)).strip()


def lines_to_pairs(lines: list[tuple[float, str]]) -> list[dict]:
    """Convert annotated lines into ordered Q/A pairs."""
    pairs: list[dict] = []
    current_q: list[str] = []
    current_a: list[str] = []
    in_qa = False  # True once we've seen question 1

    for size, text in lines:
        is_question_size = size <= QUESTION_SIZE_MAX
        starts_new_question = bool(re.match(r"^\d+\.", text)) and is_question_size
        starts_first_question = bool(re.match(r"^1\.", text)) and is_question_size

        if starts_first_question and not in_qa:
            in_qa = True
            current_q = [text]
            current_a = []
        elif starts_new_question and in_qa:
            if current_a:
                # Close previous pair — answer text has been collected
                pairs.append(
                    {
                        "question": join_lines(current_q),
                        "answer": join_lines(current_a),
                    }
                )
                current_q = [text]
                current_a = []
            else:
                # No answer yet — questions are combined (e.g. "Fragen 1 und 2 gemeinsam")
                current_q.append(text)
        elif in_qa and is_question_size:
            current_q.append(text)
        elif in_qa and not is_question_size:
            current_a.append(text)

    # Flush last pair
    if current_q and current_a:
        pairs.append(
            {
                "question": join_lines(current_q),
                "answer": join_lines(current_a),
            }
        )

    return pairs


def extract_index(index_path: Path, output_path: Path) -> None:
    pdf_dir = index_path.parent / "pdfs"
    skipped = 0
    written = 0

    with (
        open(index_path, encoding="utf-8") as idx,
        open(output_path, "w", encoding="utf-8") as out,
    ):
        for line in idx:
            record = json.loads(line)
            pdf_file = record["pdf_file"]
            vorgang_id = record["vorgang"]["id"]
            pdf_path = pdf_dir / pdf_file

            if not pdf_path.exists():
                print(f"  MISSING: {pdf_file}")
                skipped += 1
                continue

            lines = extract_lines(pdf_path)
            if lines is None:
                print(f"  NO TEXT LAYER: {pdf_file} — skipping")
                skipped += 1
                continue

            pairs = lines_to_pairs(lines)
            if not pairs:
                print(f"  NO PAIRS: {pdf_file} — skipping")
                skipped += 1
                continue

            out.write(
                json.dumps(
                    {"pdf_file": pdf_file, "vorgang_id": vorgang_id, "pairs": pairs},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            print(f"  {pdf_file}: {len(pairs)} pairs", end="\r")

    print(f"\nDone. Written {written} records, skipped {skipped}.")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Q/A pairs from Antwort PDFs listed in a fetch index."
    )
    parser.add_argument("index", metavar="INDEX", help="Path to the index .jsonl file")
    parser.add_argument("output", metavar="OUTPUT", help="Output .jsonl file path")
    args = parser.parse_args()

    index_path = Path(args.index)
    output_path = Path(args.output)

    extract_index(index_path, output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Delete an index file and its associated PDF files."""

import argparse
import json
from pathlib import Path


def delete_index(index_path: Path, *, dry_run: bool = False) -> None:
    if not index_path.exists():
        print(f"Index not found: {index_path}")
        return

    pdf_dir = index_path.parent / "pdfs"
    deleted_pdfs = 0
    missing_pdfs = 0

    with open(index_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pdf_file = record.get("pdf_file")
            if not pdf_file:
                continue
            pdf_path = pdf_dir / pdf_file
            if pdf_path.exists():
                if not dry_run:
                    pdf_path.unlink()
                deleted_pdfs += 1
            else:
                missing_pdfs += 1

    if not dry_run:
        index_path.unlink()

    action = "Would delete" if dry_run else "Deleted"
    print(f"{action} {deleted_pdfs} PDF(s).")
    if missing_pdfs:
        print(f"  {missing_pdfs} PDF(s) already missing.")
    if not dry_run:
        print(f"Deleted index: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete an index file and its associated PDF files."
    )
    parser.add_argument("index", metavar="INDEX", help="Path to the index .jsonl file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    args = parser.parse_args()

    delete_index(Path(args.index), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

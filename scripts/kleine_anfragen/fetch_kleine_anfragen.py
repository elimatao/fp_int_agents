#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx"]
# ///
"""Fetch Antworten auf Kleine Anfragen from the DIP Bundestag API.

Output structure:
  <output_dir>/
    index.jsonl   — one record per Antwort: vorgang metadata + pdf filename
    pdfs/         — one PDF per Antwort, named <dokumentnummer>.pdf
"""

import argparse
import json
import time
import tomllib
from pathlib import Path

import httpx

BASE_URL = "https://search.dip.bundestag.de/api/v1"
REQUEST_DELAY = 0.5  # seconds between requests
VORGANG_PAGE_SIZE = 100
VORGANGSPOSITION_PAGE_SIZE = 100


def load_api_key() -> str:
    config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config["API_KEY"]


def api_get(client: httpx.Client, path: str, params: dict) -> dict:
    response = client.get(BASE_URL + path, params=params)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.json()


def fetch_antwort_id(client: httpx.Client, vorgang_id: str) -> str | None:
    """Return the Drucksache ID of the Antwort for a Vorgang, or None."""
    params = {
        "f.vorgang": vorgang_id,
        "f.dokumentart": "Drucksache",
        "format": "json",
    }
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor
        data = api_get(client, "/vorgangsposition", params)
        for pos in data.get("documents", []):
            fundstelle = pos.get("fundstelle", {})
            if fundstelle.get("drucksachetyp") == "Antwort":
                return str(fundstelle["id"])

        cursor = data.get("cursor")
        if not cursor or len(data.get("documents", [])) < VORGANGSPOSITION_PAGE_SIZE:
            break

    return None


def fetch_drucksache(client: httpx.Client, doc_id: str) -> dict:
    return api_get(client, f"/drucksache/{doc_id}", {"format": "json"})


def download_pdf(client: httpx.Client, url: str, dest: Path) -> bool:
    """Download PDF to dest. Returns False if the file is not yet available (404)."""
    response = client.get(url)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    dest.write_bytes(response.content)
    time.sleep(REQUEST_DELAY)
    return True


def _index_filename(
    *,
    wahlperiode: list[int] | None,
    date_start: str | None,
    date_end: str | None,
) -> str:
    parts = ["index"]
    if wahlperiode:
        parts.append("wp" + "-".join(str(w) for w in wahlperiode))
    if date_start:
        parts.append(date_start)
    if date_end:
        parts.append(date_end)
    return "_".join(parts) + ".jsonl"


def fetch_kleine_anfragen(
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    wahlperiode: list[int] | None = None,
    max_results: int | None = None,
    output_dir: str,
) -> None:
    api_key = load_api_key()
    auth = {"Authorization": f"ApiKey {api_key}"}

    out_path = Path(output_dir)
    pdf_dir = out_path / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_path / _index_filename(
        wahlperiode=wahlperiode, date_start=date_start, date_end=date_end
    )

    vorgang_params: dict = {
        "f.vorgangstyp": "Kleine Anfrage",
        "format": "json",
    }
    if date_start:
        vorgang_params["f.datum.start"] = date_start
    if date_end:
        vorgang_params["f.datum.end"] = date_end
    if wahlperiode:
        vorgang_params["f.wahlperiode"] = wahlperiode

    cursor: str | None = None
    total: int | None = None
    emitted = 0
    scanned = 0

    # Load already-indexed pdf filenames to skip re-fetching
    already_fetched: set[str] = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    already_fetched.add(json.loads(line)["pdf_file"])
                except (json.JSONDecodeError, KeyError):
                    pass
        emitted = len(already_fetched)
        print(f"Resuming: {emitted} records already in index.")

    with (
        open(index_path, "a", encoding="utf-8") as index,
        httpx.Client(timeout=60, headers=auth) as client,
    ):
        while True:
            if cursor:
                vorgang_params["cursor"] = cursor

            data = api_get(client, "/vorgang", vorgang_params)

            if total is None:
                total = data.get("numFound", 0)
                print(f"Found {total} Vorgänge of type 'Kleine Anfrage'.")

            vorgaenge = data.get("documents", [])

            for vorgang in vorgaenge:
                scanned += 1
                vorgang_id = vorgang["id"]
                print(
                    f"  [{scanned}/{total}] Vorgang {vorgang_id} — checking positions ...",
                    end="\r",
                )

                antwort_id = fetch_antwort_id(client, vorgang_id)
                if not antwort_id:
                    continue

                antwort = fetch_drucksache(client, antwort_id)
                pdf_url = antwort.get("fundstelle", {}).get("pdf_url")
                if not pdf_url:
                    continue

                doc_nr = antwort.get("dokumentnummer", antwort_id).replace("/", "-")
                pdf_filename = f"{doc_nr}.pdf"
                if pdf_filename in already_fetched:
                    continue
                if not download_pdf(client, pdf_url, pdf_dir / pdf_filename):
                    print(f"  [{scanned}/{total}] PDF not yet available: {pdf_url}")
                    continue

                record = {
                    "vorgang": vorgang,
                    "antwort": antwort,
                    "pdf_file": pdf_filename,
                }
                index.write(json.dumps(record, ensure_ascii=False) + "\n")

                emitted += 1
                print(
                    f"  [{scanned}/{total}] Saved {emitted} Antworten so far.        "
                )

                if max_results and emitted >= max_results:
                    break

            if max_results and emitted >= max_results:
                break

            cursor = data.get("cursor")
            if not cursor or len(vorgaenge) < VORGANG_PAGE_SIZE:
                break

    print(f"\nDone. Scanned {scanned} Vorgänge, saved {emitted} Antworten.")
    print(f"Index: {index_path}")
    print(f"PDFs:  {pdf_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Antworten auf Kleine Anfragen from the DIP Bundestag API."
    )
    parser.add_argument(
        "--date-start", metavar="YYYY-MM-DD", help="Earliest publication date"
    )
    parser.add_argument(
        "--date-end", metavar="YYYY-MM-DD", help="Latest publication date"
    )
    parser.add_argument(
        "--wahlperiode",
        type=int,
        nargs="+",
        metavar="N",
        help="Legislative period(s), e.g. --wahlperiode 19 20",
    )
    parser.add_argument(
        "--max",
        type=int,
        dest="max_results",
        metavar="N",
        help="Maximum number of Antworten to fetch",
    )
    parser.add_argument(
        "--output",
        required=False,
        metavar="DIR",
        default="data",
        help="Output directory, defaults to data/ (will be created if needed)",
    )
    args = parser.parse_args()

    fetch_kleine_anfragen(
        date_start=args.date_start,
        date_end=args.date_end,
        wahlperiode=args.wahlperiode,
        max_results=args.max_results,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

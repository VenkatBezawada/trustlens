"""
Ingest annual-report PDFs into chunks with STABLE, DETERMINISTIC IDs.

Why stable IDs matter: the golden eval set (next step) references gold_chunk_ids.
Those IDs must not change when you re-run ingestion, or your answer key rots.
ID format:  {ticker}_{fy}_{section}_p{page:04d}_c{idx:02d}
e.g.        INFY_FY24_mda_p0112_c03

Day-1 dependency: pymupdf only.

Usage:
    python -m trustlens.ingest --manifest data/manifest.json --out data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf  # PyMuPDF (the `fitz` alias is deprecated)


# ~800 tokens ≈ ~3200 chars. Char-based keeps Day 1 dependency-free (no tokenizer).
# Swap to a real tokenizer later if you want token-exact chunks.
CHUNK_CHARS = 3200
OVERLAP_CHARS = 400


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Sliding-window char chunker with overlap. Prefers to break on paragraph
    boundaries near the window edge so chunks don't split mid-sentence."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # try to end on a paragraph/sentence boundary within the last 400 chars
            window = text[end - 400:end]
            for sep in ("\n\n", ". ", "\n"):
                pos = window.rfind(sep)
                if pos != -1:
                    end = (end - 400) + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks


def extract_pages(doc: pymupdf.Document, page_range: list[int] | None) -> list[tuple[int, str]]:
    """Return [(1-indexed page number, page text)]. page_range is [start, end] inclusive,
    1-indexed. If None, extract the whole document."""
    if page_range is None:
        pages = range(1, doc.page_count + 1)
    else:
        start, end = page_range
        pages = range(start, min(end, doc.page_count) + 1)
    out = []
    for pno in pages:
        page = doc.load_page(pno - 1)  # fitz is 0-indexed
        out.append((pno, page.get_text("text")))
    return out


def ingest_report(entry: dict) -> list[dict]:
    """Turn one manifest entry into a list of chunk records."""
    pdf_path = Path(entry["pdf"])
    if not pdf_path.exists():
        print(f"  ! missing PDF, skipping: {pdf_path}")
        return []

    doc = pymupdf.open(pdf_path)
    # If no sections given, treat the whole doc as one 'full' section.
    sections = entry.get("sections") or {"full": None}

    records: list[dict] = []
    for section, page_range in sections.items():
        for page_no, page_text in extract_pages(doc, page_range):
            for idx, chunk in enumerate(chunk_text(page_text)):
                chunk_id = f"{entry['ticker']}_{entry['fy']}_{section}_p{page_no:04d}_c{idx:02d}"
                records.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "ticker": entry["ticker"],
                    "company": entry.get("company", entry["ticker"]),
                    "fy": entry["fy"],
                    "section": section,
                    "page": page_no,
                })
    doc.close()
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as f:
        for entry in manifest:
            print(f"Ingesting {entry['ticker']} {entry['fy']} ...")
            recs = ingest_report(entry)
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(recs)
            print(f"  -> {len(recs)} chunks")

    print(f"\nDone. {total} chunks -> {out_path}")
    print("Eyeball a few, then write your golden set against these chunk_ids.")


if __name__ == "__main__":
    main()

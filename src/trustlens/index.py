"""
Build the two indexes retrieve.py needs:
  - DENSE:  embed every chunk with bge-small, store in a persistent Chroma collection.
  - SPARSE: tokenize every chunk, store a BM25Okapi index (pickled) — catches exact
            identifiers (company names, FY years, %s) that dense embeddings often fumble.

Both are built from the same chunks.jsonl, so retrieve.py can fuse the two rankings.

Usage:
    python -m trustlens.index --chunks data/processed/chunks.jsonl --chroma-dir .chroma --bm25-out data/processed/bm25.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "trustlens_chunks"

# simple whitespace+lowercase tokenizer for BM25 — good enough for company names,
# fiscal years, and numbers, which is exactly what BM25 needs to catch here.
_TOKEN_RE = re.compile(r"[a-z0-9%.]+")


def bm25_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_dense_index(chunks: list[dict], chroma_dir: Path) -> None:
    print(f"Building dense index ({EMBED_MODEL}) -> {chroma_dir}")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    # start clean each run so the index always reflects the current chunks.jsonl exactly
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"ticker": c["ticker"], "company": c["company"], "fy": c["fy"],
         "section": c["section"], "page": c["page"]}
        for c in chunks
    ]

    # bge models expect a "passage" prefix convention for asymmetric search quality;
    # for bge-small-en-v1.5 no special document prefix is required (only queries need
    # the "Represent this sentence..." instruction) — see model card.
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Chroma has a max batch add size; chunk the add() calls to be safe at scale.
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )
    print(f"  -> {collection.count()} vectors in Chroma collection '{COLLECTION_NAME}'")


def build_sparse_index(chunks: list[dict], out_path: Path) -> None:
    print(f"Building sparse (BM25) index -> {out_path}")
    tokenized = [bm25_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],  # positional alignment with bm25's internal corpus order
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"  -> BM25 over {len(tokenized)} chunks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--chroma-dir", default=".chroma")
    ap.add_argument("--bm25-out", default="data/processed/bm25.pkl")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.chunks))
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    build_dense_index(chunks, Path(args.chroma_dir))
    build_sparse_index(chunks, Path(args.bm25_out))
    print("\nDone. Both indexes are ready for retrieve.py.")


if __name__ == "__main__":
    main()

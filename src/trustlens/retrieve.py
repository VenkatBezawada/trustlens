"""
Retrieval, in two switchable modes — this is the A/B the whole eval story rests on.

  NAIVE:  dense-only top-k. What most tutorial RAGs ship.
  HYBRID: BM25 + dense candidates -> fuse with Reciprocal Rank Fusion -> cross-encoder
          rerank -> top-k. Catches what dense embeddings alone miss (exact company
          names, FY years, %-figures) and re-orders by actual query-document relevance
          instead of raw fused rank.

Both modes return the same shape so eval/run_eval.py can score them identically:
    [{"chunk_id": ..., "text": ..., "score": ..., "ticker": ..., "company": ..., "fy": ..., "section": ..., "page": ...}, ...]

Usage (CLI smoke test):
    python -m trustlens.retrieve --query "..." --mode hybrid --k 6
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from trustlens.index import COLLECTION_NAME, EMBED_MODEL, bm25_tokenize

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hybrid over-retrieves this many candidates from EACH of dense/sparse before fusing,
# then reranks and prunes down to k. Over-retrieve-then-rerank beats retrieving k directly.
OVER_RETRIEVE_N = 15

# RRF constant — standard default from the original RRF paper (Cormack et al.).
# Lower k weights top ranks more heavily; 60 is the well-established default.
RRF_K = 60


class Retriever:
    """Loads both indexes once; call .retrieve(query, mode, k) as many times as needed."""

    def __init__(self, chroma_dir: str = ".chroma", bm25_path: str = "data/processed/bm25.pkl"):
        self._embed_model = SentenceTransformer(EMBED_MODEL)
        self._reranker: CrossEncoder | None = None  # lazy-loaded, only needed for hybrid

        client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = client.get_collection(COLLECTION_NAME)

        with open(bm25_path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_chunk_ids = payload["chunk_ids"]  # positional order matches BM25's internal corpus

        # metadata lookup so both retrieval paths can return full records by chunk_id
        all_rows = self._collection.get(include=["documents", "metadatas"])
        self._by_id = {
            cid: {"text": doc, **meta}
            for cid, doc, meta in zip(all_rows["ids"], all_rows["documents"], all_rows["metadatas"])
        }

    # ---------- dense ----------
    def _dense_search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, similarity_score), ...] ranked best-first."""
        q_emb = self._embed_model.encode([query]).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=n)
        ids = res["ids"][0]
        # chroma returns squared-L2 distance by default; convert to a similarity-like score (smaller distance = higher score)
        distances = res["distances"][0]
        scores = [1.0 / (1.0 + d) for d in distances]
        return list(zip(ids, scores))

    # ---------- sparse ----------
    def _sparse_search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score), ...] ranked best-first."""
        tokens = bm25_tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._bm25_chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    # ---------- fusion ----------
    @staticmethod
    def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
        """rankings: list of ranked chunk_id lists (best-first), one per retrieval method.
        RRF score for a doc = sum over methods of 1 / (k + rank), rank is 1-indexed.
        Docs absent from a ranking simply don't contribute a term for it."""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ---------- rerank ----------
    def _rerank(self, query: str, chunk_ids: list[str]) -> list[tuple[str, float]]:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        pairs = [(query, self._by_id[cid]["text"]) for cid in chunk_ids]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked

    # ---------- public API ----------
    def retrieve(self, query: str, mode: str = "hybrid", k: int = 6) -> list[dict]:
        if mode == "naive":
            hits = self._dense_search(query, k)
        elif mode == "hybrid":
            dense_hits = self._dense_search(query, OVER_RETRIEVE_N)
            sparse_hits = self._sparse_search(query, OVER_RETRIEVE_N)
            dense_ranking = [cid for cid, _ in dense_hits]
            sparse_ranking = [cid for cid, _ in sparse_hits]
            fused = self._reciprocal_rank_fusion([dense_ranking, sparse_ranking])
            candidate_ids = [cid for cid, _ in fused[:OVER_RETRIEVE_N]]
            hits = self._rerank(query, candidate_ids)[:k]
        else:
            raise ValueError(f"unknown mode: {mode!r} (use 'naive' or 'hybrid')")

        results = []
        for cid, score in hits:
            rec = {"chunk_id": cid, "score": float(score), **self._by_id[cid]}
            results.append(rec)
        return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--mode", choices=["naive", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--chroma-dir", default=".chroma")
    ap.add_argument("--bm25-path", default="data/processed/bm25.pkl")
    args = ap.parse_args()

    retriever = Retriever(chroma_dir=args.chroma_dir, bm25_path=args.bm25_path)
    hits = retriever.retrieve(args.query, mode=args.mode, k=args.k)

    print(f"\nQuery: {args.query}\nMode: {args.mode}\n")
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["text"].split())[:120]
        print(f"{i}. [{h['score']:.4f}] {h['chunk_id']} ({h['company']})\n   {snippet}...\n")


if __name__ == "__main__":
    main()

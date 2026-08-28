# TrustLens

**An eval-driven RAG system over Indian company annual reports, built so retrieval quality is measured — not assumed — and a CI gate blocks any change that regresses it.**

[![RAG Eval Gate](https://github.com/VenkatBezawada/trustlens/actions/workflows/eval.yml/badge.svg)](https://github.com/VenkatBezawada/trustlens/actions/workflows/eval.yml)

Most RAG portfolios stop at "it retrieves and answers." This one is built around the opposite question: **how do you know it's actually retrieving the right thing, and how do you stop a regression from shipping silently?**

---

## The result, before anything else

Two retrieval strategies, evaluated on the same 16-question golden set with exact source citations required:

| metric | naive (dense-only) | hybrid (BM25 + dense + rerank) |
|---|---:|---:|
| context recall | 76.2% | **77.4%** |
| citation ↔ gold-source overlap | 76.2% | **77.4%** |
| abstention accuracy | 81.2% | **93.8%** |
| **fair success rate*** | 64.3% | **71.4%** |
| CI gate (80% recall / 90% success) | — | ❌ fails |

*\*"Fair success rate" isn't a standard metric — I built it after finding that the obvious one (accuracy on questions the system chose to answer) was misleading. See below.*

Hybrid wins on every axis, by a real and reproducible margin. Neither mode clears production quality yet — the CI gate says so honestly, and the root causes are diagnosed, not mysterious (see **Findings**).

---

## Why this exists

Production RAG doesn't fail where tutorials show it succeeding — it fails at retrieval, silently, while generation stays fluent and confident. A system can score high on "does the answer sound coherent" while quietly answering from the wrong document. Most portfolio RAG projects never measure that gap because they never had a way to.

This project is built the other way around: **the golden answer set was written before the retrieval pipeline**, so "correct" was defined first, and every retrieval/generation decision was built to be scored against it — not the other way around.

## Architecture

```
PDFs (NSE/company IR)  →  narrative-section extraction (PyMuPDF)
                        →  chunking, stable deterministic IDs (hand-rolled, no LangChain)
                        →  dense index (bge-small-en-v1.5 + Chroma) + sparse index (BM25)

query  →  [naive: dense top-k]  or  [hybrid: dense+BM25 → RRF fusion → cross-encoder rerank]
       →  generation with citation enforcement (OpenAI bracket-citations, or Anthropic's
          structural citations API as an alt backend — see generate.py for the tradeoff)
       →  answer + resolved source chunk_ids

eval  →  16-question golden set (single-hop / multi-hop / deliberately unanswerable)
      →  deterministic retrieval metrics (chunk-ID overlap, no LLM judge needed)
      →  LLM-judged correctness (gpt-5-mini, ~$0.05 for a full 32-call sweep)
      →  CI gate — fails the build on a real regression, not a vibe
```

Corpus: FY24 MD&A / narrative sections from Sun Pharma, HDFC Bank, and Cipla (188 chunks total), sourced from NSE's public archive and company investor-relations pages. Financial tables are explicitly out of scope for this phase — narrative text only.

## Findings (the part that's actually worth reading)

**1. The obvious accuracy metric was lying, and I caught it by checking the denominator, not the headline number.**
Naive scored 81.8% "correctness" vs. hybrid's 76.9% — naive looked better. But that number is only computed over questions each mode *chose to answer*. Naive wrongfully abstained on 3 questions hybrid answered correctly. Once abstention is scored as a failure (same as a wrong answer — the user gets nothing either way) on a fixed denominator, hybrid wins **71.4% to 64.3%**. The eval harness (`eval/run_eval.py`) now computes this fair metric automatically; the misleading one is still shown too, labeled explicitly as denominator-biased, so the discrepancy is visible rather than hidden.

**2. A real chunking bug, traced to the exact character in the exact PDF page.**
Both retrieval modes missed the canonical source for a Cipla R&D figure. Root cause, found by reading the raw extracted text: page 67 is a dense, punctuation-free financial ratio table; the boundary-detection chunker cut mid-sentence right after the number, splitting one fact across two chunks. Fixed at the golden-set level (both valid chunks are now accepted answers); a proper fix means re-chunking, which shifts every chunk ID in the corpus — documented as the next concrete improvement, not silently patched over.

**3. One page was deliberately left unprocessed rather than guessed.**
Infosys's integrated report has no conventional "MD&A" heading — the content exists but isn't cleanly extractable with the same page-range approach used elsewhere. Rather than fabricate a page range, it's excluded from the corpus. The unanswerable-question bucket in the golden set exists specifically to test that the system says "I don't know" instead of hallucinating in cases exactly like this.

## CI

[`.github/workflows/eval.yml`](.github/workflows/eval.yml) runs on every push/PR. It's designed to need nothing from a stranger cloning the repo: if no `OPENAI_API_KEY` secret is configured, it automatically runs a free `--dry-run` that validates the whole pipeline and reports real (deterministic) retrieval metrics. Add a key as a repo secret and the same workflow upgrades itself to the real, LLM-judged gate — no code changes required. See [`GITHUB_SETUP.md`](GITHUB_SETUP.md).

## Stack

| | |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | hand-rolled (deliberately, not a library — see `ingest.py`) |
| Dense retrieval | `bge-small-en-v1.5` + Chroma |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion / rerank | Reciprocal Rank Fusion → cross-encoder (`ms-marco-MiniLM`) |
| Generation | OpenAI `gpt-5-mini` (default) or Anthropic `claude-sonnet-5` (`--backend anthropic`) |
| Eval | custom — deterministic chunk-overlap metrics + LLM-judged correctness |
| CI | GitHub Actions, key-optional |

## Running it

```bash
pip install -e .
python -m trustlens.index                                    # builds the vector + BM25 indexes
python -m eval.run_eval --backend openai --gate               # full evaluation sweep, ~$0.05-0.10
```

Full setup (including Windows/PowerShell-specific steps): [`SETUP.md`](SETUP.md).

## What's next

- Expand the golden set past 16 questions — small enough that a couple of judge-call variances move the aggregate a few points.
- Re-chunk with a smaller window so dense financial tables stop diluting individual chunk embeddings; re-validate the full golden set against the new chunk IDs.
- Phase 2: an injection-defense layer with a red-team eval harness, using this same measurement scaffolding to prove attack resistance with real before/after numbers, not just claims.

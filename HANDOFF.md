# TrustLens — Project Handoff

*Eval-driven RAG over Indian annual reports. Everything below is what's been decided and built so far — paste this whole doc as context when you continue elsewhere.*

---

## 1. The idea, and why

Source: a YouTube video ("5 AI Engineer Projects to Build in 2026," Aishwarya Srinivasan) recommending 5 portfolio projects, RAG being #1. Verdict on the video: correct instinct (stop building demos, show production concerns) but the project list is generic — same 5 things every bootcamp grad builds. Decision: don't build a vanilla RAG chatbot. Build the thing almost nobody puts in a portfolio: **a RAG system where retrieval quality is measured, not assumed** — an eval/observability harness, built *eval-first*.

**Why eval-first, specifically:** research into 2026 production RAG failures found the dominant failure mode is a system that scores well on generation metrics (faithfulness) while silently failing on retrieval (context recall) — nobody catches it because dashboards only track generation. Building the golden answer-key first, then building retrieval to pass it, is the correct engineering order and a stronger interview story than "I built a chatbot."

**Two-phase plan:**
- **Phase 1 (current):** hybrid retrieval (BM25 + dense) + reranking + citation enforcement, measured against a hand-labeled golden set, gated in CI so a metric regression can't ship.
- **Phase 2 (later):** add an injection-defense / red-team layer on top — provable via the Phase 1 measurement scaffolding ("attack success 90% → X% after my defense").

**One-line pitch for the README:**
> "I built a RAG system over Indian company annual reports where retrieval quality is measured on a golden set, and a CI gate blocks any change that drops recall or faithfulness below threshold."

---

## 2. Corpus decision

**Indian listed-company annual reports**, downloaded free from NSE's public archive (`nsearchives.nseindia.com`) — the desi analog of a US SEC 10-K. Chosen over the 10-K default because: (a) motivation — the person wanted an India-relevant corpus, (b) role relevance — same entity-collision / multi-hop failure modes that make eval interesting, (c) free, no login, no licensing friction.

**Scope: narrative sections only** (Management Discussion & Analysis, Directors'/Board's Report) — financial tables are a known extraction headache in Indian AR formats and are explicitly deferred as a stretch goal, not attempted in Phase 1.

**Companies ingested so far:**
| Ticker | Company | FY | Section | Pages | PDF source |
|---|---|---|---|---|---|
| SUNPHARMA | Sun Pharmaceutical Industries | FY24 | MD&A | 17–67 | nsearchives.nseindia.com/annual_reports/AR_24355_SUNPHARMA_2023_2024_08072024153247.pdf |
| HDFCBANK | HDFC Bank | FY24 | MD&A | 241–277 | nsearchives.nseindia.com/annual_reports/AR_24576_HDFCBANK_2023_2024_18072024183453.pdf |
| CIPLA | Cipla | FY24 | MD&A (integrated-report style, titled "Financial Capital and Management Discussion and Analysis") | 64–73 | cipla.com/sites/default/files/Annual-Report-2023-24-(Double%20page).pdf |

**Total: 188 chunks** (86 Sun Pharma + 74 HDFC Bank + 28 Cipla).

**Infosys — parked, not ingested.** Its integrated report (`AR_24073_INFY_2023_2024_03062024153021.pdf`, 353 pages) has no conventional "MD&A" heading; the narrative content is woven into differently-named strategy sections. Rather than guess a page range, this was explicitly left undone. Cipla's report used the same integrated-reporting style but was successfully bounded (see below) — Infosys could likely be resolved the same way if revisited.

**Cipla ingestion — a worked example of the manual-boundary discipline paying off:** the heading "Financial Capital and Management Discussion and Analysis" appears at p64, but a naive heading search also flagged p74 as a candidate (a false positive — that page just *references* MD&A in passing, inside the formal Board's Report). Verified by reading actual page content: p64–73 is genuine narrative (regional business commentary, R&D spend, risk factors); p74 onward is Board's Report boilerplate (dividend policy, CSR annexures, FEMA compliance). Bounded at p73 based on content, not heading pattern-matching alone.

**Same-sector multi-hop — now real, not just disambiguation.** Added 5 new golden questions (q012–q016) after mining Cipla's actual numbers: 3 single-hop (R&D spend +17%, North America revenue USD 906M/+24% YoY, operating margin 20.3%) and 2 genuine synthesis multi-hop comparing real figures across two pharma companies' own MD&A sections — e.g. Sun Pharma's US business (32% of consolidated revenue) vs Cipla's North America business (30% of topline) for the same FY. One question (q016) deliberately tests an *asymmetric* disclosure case: Cipla states a specific R&D growth %, Sun Pharma's MD&A discusses R&D capabilities but doesn't quantify spend growth — a realistic "which source actually answers this" retrieval test.

---

## 3. What's built (files + what each does)

Repo root: `trustlens/`

```
trustlens/
├── pyproject.toml              # deps, phased/commented — Day 1 needs only pymupdf
├── .gitignore
├── data/
│   ├── manifest.json           # per-company PDF path + narrative page ranges (2 entries, real & verified)
│   ├── raw/                    # the 3 downloaded PDFs (Infosys, Sun Pharma, HDFC — Infosys unused)
│   ├── processed/
│   │   └── chunks.jsonl        # 160 chunks — 86 Sun Pharma + 74 HDFC Bank
│   └── golden/
│       └── golden_set.jsonl    # 11 hand-labeled QA pairs (the answer key)
├── src/trustlens/
│   └── ingest.py               # PDF -> narrative text -> chunks with STABLE deterministic IDs
└── eval/
    └── load_golden.py          # loads + validates golden set against chunks.jsonl
```

### `ingest.py` — PDF extraction + chunking
- Tool used: **PyMuPDF** (`pymupdf`, the `fitz` import is deprecated — use `import pymupdf`) for text extraction only.
- Chunking is **hand-rolled**, not a library (no LangChain/LlamaIndex splitter) — deliberate, to keep Day-1 dependencies at zero and to be able to explain the mechanics in an interview.
- Chunking algorithm: sliding window, `CHUNK_CHARS = 3200` (~800 tokens, character-based not token-based — a known simplification, flagged for later upgrade to a real tokenizer), `OVERLAP_CHARS = 400`. Before cutting a window, it looks back up to 400 chars for a clean break point — tries `\n\n` (paragraph) first, then `. ` (sentence), then `\n` (line) — so chunks don't split mid-sentence. Next chunk starts 400 chars before the previous one ended, so context overlaps the boundary.
- **Chunk ID format (the important design decision):** `{ticker}_{fy}_{section}_p{page:04d}_c{idx:02d}`, e.g. `SUNPHARMA_FY24_mda_p0017_c00`. Deterministic and stable across re-runs — critical because the golden set references these IDs directly; if IDs changed on re-ingest, the answer key would silently rot.
- Driven by `manifest.json` — one entry per company with `pdf` path and `sections: {section_name: [start_page, end_page]}` (1-indexed, inclusive). Manual page ranges were chosen over auto-detecting section headings — auto-detection was tested (font-size + regex heuristics) and proved too unreliable across the heterogeneous Indian AR layouts (confirmed empirically: worked cleanly for Sun Pharma, failed for Infosys/HDFC). Manual ranges cost ~5 min/company but are reliable.
- Verified end-to-end: ran on real PDFs, produced 160 chunks with correct stable IDs.

### `golden_set.jsonl` — the answer key
**16 QA pairs**, JSONL, one record per line. Schema:
```json
{"id": "q001", "question": "...", "answer": "...", "gold_chunk_ids": ["SUNPHARMA_FY24_mda_p0017_c00"], "type": "single_hop", "companies": ["SUNPHARMA"], "fy": "FY24"}
```
Three buckets present:
- **10 single-hop** — real, checkable facts mined directly from chunk text (e.g., India FY24 real GDP growth 8.2% per HDFC's MD&A; Sun Pharma protected-brands market share 48%→54%; HDFC Other Income grew 57.7% to ₹49,241.0 crore; Cipla R&D spend +17%, North America revenue USD 906M/+24% YoY, operating margin 20.3%). Every answer traces to a real `gold_chunk_id` — nothing fabricated.
- **4 multi-hop** — two flavors, deliberately:
  - **Cross-sector disambiguation** (q008, q009 — Sun Pharma vs HDFC): tests whether retrieval correctly attributes a fact to the right company and doesn't blend figures across unrelated companies. Easier — sectors are obviously different.
  - **Same-sector synthesis** (q015, q016 — Sun Pharma vs Cipla, both pharma): the harder, more realistic case. q015 compares two real same-metric figures across two documents (US/North America revenue contribution: Sun Pharma 32% vs Cipla 30%). q016 tests an *asymmetric* disclosure — only Cipla states a specific R&D growth %, so the correct answer requires knowing when a source does or doesn't contain the requested fact, not just retrieving the nearest match.
- **2 unanswerable** — "What was Infosys's FY24 revenue?" / "What was TCS's FY24 revenue?" (q011 originally asked about Cipla's R&D spend, but that became answerable once Cipla was added to the corpus — caught and fixed by re-running the validator after every ingestion change, which is exactly the discipline this project is meant to demonstrate). Correct system behavior is to **abstain**, not hallucinate. This bucket is the one most portfolios skip.

Target before trusting the eval numbers: **~30–50 questions total** (16 is a validated, now-multi-sector starter, not final).

### `load_golden.py` — the validator
Loads the golden set and checks: required fields present, no duplicate IDs, `type` is one of `single_hop|multi_hop|unanswerable`, unanswerable questions have empty `gold_chunk_ids`, answerable questions have non-empty `gold_chunk_ids`, and — the critical check — **every `gold_chunk_id` referenced actually exists in `chunks.jsonl`**. Exits non-zero on any problem (built to become a CI gate later). Tested both ways: clean pass on the real golden set, correctly caught and failed (exit code 1) on a deliberately injected typo'd chunk ID.

---

## 4. Full architecture (target — not all built yet)

```
                        INGESTION (✓ built)
  PDFs (NSE archive) ──▶ extract narrative text ──▶ chunk + stable IDs ──▶ chunks.jsonl
                          (PyMuPDF)                  (hand-rolled splitter)

                        INDEXING (not built yet)
  chunks.jsonl ──▶ dense: embeddings → vector store (Chroma)
                └─▶ sparse: BM25 index

                        RETRIEVAL + GENERATION (not built yet)
  question ──▶ hybrid retrieve (dense+BM25) ──▶ fuse (RRF) ──▶ rerank (cross-encoder) ──▶ Claude w/ citations ──▶ answer

                        EVALUATION (partially built — golden set ✓, runner not built)
  golden_set.jsonl ──▶ run pipeline ──▶ RAGAS metrics (context_recall, context_precision, faithfulness, answer_relevancy) ──▶ pass/fail

                        OBSERVABILITY (not built yet)
  Langfuse traces — p50/p95 latency, cost/request

                        CI (not built yet)
  GitHub Actions — fails build if context_recall < 0.80 or faithfulness < 0.90
```

### Tech stack decided (with reasoning)
| Layer | Choice | Why |
|---|---|---|
| PDF→text | PyMuPDF | fast, clean, already proven to work |
| Dense embeddings | `BAAI/bge-small-en-v1.5` (local) | free, reproducible |
| Vector store | Chroma | fastest to ship, persists locally |
| Sparse | BM25 (`rank_bm25`) | catches exact identifiers/company names dense search fumbles |
| Fusion | Reciprocal Rank Fusion | simple, combines dense+sparse rankings |
| Reranker | cross-encoder (`ms-marco-MiniLM-L-6-v2` or `bge-reranker-base`) | local, free; over-retrieve ~15 → prune to ~6 |
| Generator | Claude API, with citations | citations map claims→source spans, satisfying the citation-enforcement requirement for free |
| Eval | RAGAS | needs an LLM judge (Claude or Gemini) |
| Observability | Langfuse | traces, latency, cost dashboards |
| CI | GitHub Actions | gate on metric regression |

### The core eval insight (don't lose this)
Track **both** a retrieval-stage metric (context_recall — did we fetch the right chunks) and a generation-stage metric (faithfulness — did the answer stay grounded in what was fetched), always together. A system can score high faithfulness while quietly failing recall — the generator writes a confident, coherent answer from *partial* context, and a generation-only dashboard never sees it. This is the specific failure the whole project is built to catch and demonstrate.

---

## 5. Definition of done (Phase 1)
README shows: (a) naive-vs-hybrid retrieval numbers on the golden set, (b) one multi-hop question naive fails and hybrid+rerank passes, (c) a latency/cost dashboard screenshot, (d) a CI run screenshot that blocks a deliberately-broken change (e.g. a PR that drops the reranker or sets k=1).

---

## 6. Immediate next steps, in order
1. ~~Add a third, same-sector company~~ **✓ DONE** — Cipla ingested (28 chunks, p64–73), 5 new golden questions added including 2 genuine same-sector synthesis multi-hop pairs. Golden set now 16 questions, revalidated clean.
2. **Continue expanding the golden set** from 16 → ~30–50 questions (16 is a good working starter, not final).
3. **Build `index.py`** — embed chunks into Chroma (dense) + build a BM25 index (sparse).
4. **Build `retrieve.py`** — naive top-k mode AND hybrid (BM25+dense → RRF → cross-encoder rerank) mode, switchable, so naive-vs-hybrid can be A/B'd on the golden set.
5. **Build `generate.py`** — Claude call with citation enforcement (claims mapped to source chunk spans).
6. **Build `eval/run_eval.py`** — run the pipeline (both modes) over `golden_set.jsonl`, score with RAGAS, produce the before/after report.
7. **Wire Langfuse** for traces/latency/cost.
8. **Add the GitHub Actions CI gate** (fail on `context_recall < 0.80` or `faithfulness < 0.90`); demonstrate it by opening a PR that breaks retrieval and screenshotting the red check.
9. **Write the README** around the before/after numbers, the multi-hop win, the dashboard screenshot, and the blocked-regression screenshot.
10. **Phase 2** (after Phase 1 ships): injection-defense + red-team harness on top of the same measurement scaffolding.

---

## 7. Working preferences established in this thread (carry these over)
- No fabricated/estimated facts — every claim in the golden set traces to a real chunk; when something couldn't be verified (Infosys MD&A location), it was left undone rather than guessed.
- Prefer hand-rolled/explainable components over black-box libraries where the extra dependency isn't worth it (chunking done by hand, not via LangChain).
- Manual/verified over auto-detected when auto-detection proved unreliable (section page ranges).
- Ship incrementally and validate each step before moving on (smoke-tested ingest before trusting it; proved the validator catches errors before trusting it).

"""
run_eval.py — the centerpiece: runs the golden set through naive and hybrid
retrieval, generates a cited answer for each, scores both retrieval and
generation quality, and produces the before/after numbers this whole project
exists to show.

Cost note: a full sweep is 16 questions x 2 modes = 32 generate() calls plus 32
judge_correctness() calls (only for non-dry-run). At gpt-5-mini rates this is a
few cents total — see the cost breakdown discussed earlier in the project.

Retrieval metrics (context_recall, context_precision) are DETERMINISTIC and cost
nothing — they only depend on which chunk_ids retrieve() returns, never touch an
LLM. That means --dry-run still produces real, meaningful retrieval numbers; only
the generation-stage numbers (citation overlap, correctness, abstention) are fake
in dry-run, since those depend on what the model actually says.

Usage:
    python -m eval.run_eval --dry-run                    # free — real retrieval numbers, fake generation
    python -m eval.run_eval --backend openai --k 6        # real — costs a few cents
    python -m eval.run_eval --backend openai --gate       # also apply the CI thresholds and exit non-zero on fail
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import (
    abstention_correct,
    citation_gold_overlap,
    context_precision,
    context_recall,
    judge_correctness,
    _dry_run_judge_correctness,
)

# CI gate thresholds — see PLAN.md sec. 8. Applied only when --gate is passed.
MIN_CONTEXT_RECALL = 0.80
MIN_CORRECTNESS = 0.90  # fraction of answerable questions judged correct (stands in for "faithfulness")


def load_golden(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def run_single(record: dict, mode: str, backend: str, dry_run: bool, retriever, generate_fn, k: int) -> dict:
    """Retrieve + generate + score ONE golden question in ONE retrieval mode."""
    query = record["question"]
    is_answerable = record["type"] != "unanswerable"
    gold_ids = set(record["gold_chunk_ids"])

    chunks = retriever.retrieve(query, mode=mode, k=k)
    retrieved_ids = {c["chunk_id"] for c in chunks}

    result = generate_fn(query, chunks, backend=backend, dry_run=dry_run)

    recall = context_recall(retrieved_ids, gold_ids) if is_answerable else None
    precision = context_precision(retrieved_ids, gold_ids) if is_answerable else None
    cite_overlap = citation_gold_overlap(result.cited_chunk_ids, gold_ids) if is_answerable else None
    abst_ok = abstention_correct(is_answerable, result.abstained)

    if dry_run:
        is_correct, judge_text = _dry_run_judge_correctness()
    else:
        # Only worth judging correctness when the abstain/answer decision itself was
        # right — an unanswerable question the system correctly refused has no
        # "answer" to fact-check; scoring it separately via abst_ok is what matters.
        if is_answerable and not result.abstained:
            is_correct, judge_text = judge_correctness(query, record["answer"], result.answer)
        else:
            is_correct, judge_text = None, "[skipped: abstained or unanswerable — scored via abstention_correct instead]"

    return {
        "id": record["id"], "type": record["type"], "mode": mode,
        "question": query, "golden_answer": record["answer"], "model_answer": result.answer,
        "retrieved_ids": sorted(retrieved_ids), "gold_ids": sorted(gold_ids),
        "cited_ids": sorted(result.cited_chunk_ids), "abstained": result.abstained,
        "context_recall": recall, "context_precision": precision,
        "citation_gold_overlap": cite_overlap, "abstention_correct": abst_ok,
        "is_correct": is_correct, "judge_text": judge_text,
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean over non-None values only — a None means 'not applicable' (e.g. an
    unanswerable question has no gold_chunk_ids), NOT zero. Averaging Nones as
    zero would silently punish the unanswerable bucket for a metric that was
    never meant to apply to it."""
    def mean_of(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    n_answerable = sum(1 for r in rows if r["type"] != "unanswerable")
    n_unanswerable = len(rows) - n_answerable

    # answerable_success_rate: a FAIR, same-denominator metric across the full
    # answerable bucket. correctness_rate above is conditioned on "questions this
    # mode chose to answer" — that denominator shrinks for a mode that abstains
    # more, which can make a system that dodges hard questions look MORE accurate
    # than one that attempts them. A wrongful abstention costs the user just as
    # much as a wrong answer (they get nothing either way), so it's scored as a
    # failure here, not excluded. Always compare modes on THIS number, not
    # correctness_rate alone — see the project's eval writeup for a live example
    # of exactly this discrepancy changing which mode looks better.
    answerable_rows = [r for r in rows if r["type"] != "unanswerable"]
    # Distinguish "never judged" (dry-run, is_correct is None for every row) from
    # "judged and failed" (real run, is_correct is False). Without this check, a
    # dry-run would report 0.0% here — reading as "the system failed everything"
    # when the true meaning is "we never checked." Caught by actually running this
    # code and looking critically at the output, not by inspection alone.
    any_judged = any(r["is_correct"] is not None for r in answerable_rows)
    if not any_judged:
        answerable_success_rate = None
    else:
        successes = sum(1 for r in answerable_rows if not r["abstained"] and r["is_correct"] is True)
        answerable_success_rate = successes / len(answerable_rows) if answerable_rows else None

    return {
        "n_questions": len(rows),
        "n_answerable": n_answerable, "n_unanswerable": n_unanswerable,
        "mean_context_recall": mean_of("context_recall"),
        "mean_context_precision": mean_of("context_precision"),
        "mean_citation_gold_overlap": mean_of("citation_gold_overlap"),
        "abstention_accuracy": mean_of("abstention_correct"),  # applies to ALL rows, answerable + unanswerable
        "correctness_rate": mean_of("is_correct"),  # NOTE: denominator = answered-only, see answerable_success_rate
        "answerable_success_rate": answerable_success_rate,  # fair, fixed denominator = all answerable questions
    }


def print_report(naive_agg: dict, hybrid_agg: dict) -> None:
    def fmt(v):
        return f"{v:.1%}" if isinstance(v, float) else "N/A"

    rows = [
        ("context_recall (retrieval)", naive_agg["mean_context_recall"], hybrid_agg["mean_context_recall"]),
        ("context_precision (retrieval)", naive_agg["mean_context_precision"], hybrid_agg["mean_context_precision"]),
        ("citation_gold_overlap (generation)", naive_agg["mean_citation_gold_overlap"], hybrid_agg["mean_citation_gold_overlap"]),
        ("abstention_accuracy (generation)", naive_agg["abstention_accuracy"], hybrid_agg["abstention_accuracy"]),
        ("correctness_rate, answered-only (generation)", naive_agg["correctness_rate"], hybrid_agg["correctness_rate"]),
        ("answerable_success_rate, FAIR metric (generation)", naive_agg["answerable_success_rate"], hybrid_agg["answerable_success_rate"]),
    ]
    name_w = max(len(r[0]) for r in rows)
    print(f"\n{'metric':<{name_w}}  {'naive':>8}  {'hybrid':>8}")
    print("-" * (name_w + 22))
    for name, n, h in rows:
        print(f"{name:<{name_w}}  {fmt(n):>8}  {fmt(h):>8}")
    print("\nNote: correctness_rate is conditioned on questions each mode chose to")
    print("answer, so a mode that abstains more can look artificially more accurate.")
    print("answerable_success_rate uses a fixed denominator (all answerable questions,")
    print("wrongful abstention scored as a failure) and is the fairer cross-mode comparison.")


def apply_gate(hybrid_agg: dict, dry_run: bool = False) -> bool:
    """Returns True if the gate PASSES. Only checked against the hybrid mode —
    hybrid is the system we're actually shipping; naive is the baseline it's
    compared against, not something we're gating on.

    Split into two independent checks because they have different costs:
      - context_recall is DETERMINISTIC (pure chunk-ID overlap, no LLM) — it's
        real and enforceable even with zero API calls, so it's checked EVERY
        time, dry-run or not.
      - answerable_success_rate needs a real correctness judge, so it's only
        checked when real (non-dry-run) data exists. In dry-run, this half is
        reported as skipped rather than silently passed — a dry run proves
        nothing about correctness and shouldn't be able to green-light a gate
        on it.

    This means the free path (no API key configured) can still catch a real
    retrieval regression — the paid path additionally catches a correctness
    regression. Neither one can be gamed by the other being absent.
    """
    recall = hybrid_agg["mean_context_recall"]
    success = hybrid_agg["answerable_success_rate"]
    ok = True

    if recall is None or recall < MIN_CONTEXT_RECALL:
        print(f"GATE FAIL: context_recall {recall} < {MIN_CONTEXT_RECALL}")
        ok = False
    else:
        print(f"GATE PASS: context_recall {recall:.1%} >= {MIN_CONTEXT_RECALL:.0%}")

    if dry_run:
        print("GATE SKIPPED (partial): answerable_success_rate needs real judge data "
              "— not checked in dry-run. Only the free context_recall gate applies.")
    elif success is None or success < MIN_CORRECTNESS:
        print(f"GATE FAIL: answerable_success_rate {success} < {MIN_CORRECTNESS}")
        ok = False
    else:
        print(f"GATE PASS: answerable_success_rate {success:.1%} >= {MIN_CORRECTNESS:.0%}")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden_set.jsonl")
    ap.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gate", action="store_true", help="apply CI thresholds and exit non-zero on failure")
    ap.add_argument("--chroma-dir", default=".chroma")
    ap.add_argument("--bm25-path", default="data/processed/bm25.pkl")
    ap.add_argument("--out-dir", default="eval/reports")
    args = ap.parse_args()

    from trustlens.retrieve import Retriever
    from trustlens.generate import generate

    golden = load_golden(args.golden)
    retriever = Retriever(chroma_dir=args.chroma_dir, bm25_path=args.bm25_path)

    print(f"Running {len(golden)} questions x 2 modes (naive, hybrid) "
          f"{'[DRY RUN — no API calls]' if args.dry_run else f'[backend={args.backend}]'}")

    all_rows: list[dict] = []
    for mode in ("naive", "hybrid"):
        for i, record in enumerate(golden, 1):
            print(f"  [{mode:6s}] {i}/{len(golden)}: {record['id']}", end="\r")
            row = run_single(record, mode, args.backend, args.dry_run, retriever, generate, args.k)
            all_rows.append(row)
        print()

    naive_rows = [r for r in all_rows if r["mode"] == "naive"]
    hybrid_rows = [r for r in all_rows if r["mode"] == "hybrid"]
    naive_agg = aggregate(naive_rows)
    hybrid_agg = aggregate(hybrid_rows)

    print_report(naive_agg, hybrid_agg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"eval_{timestamp}{'_dryrun' if args.dry_run else ''}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp, "dry_run": args.dry_run, "backend": args.backend, "k": args.k,
        "naive": {"aggregate": naive_agg, "rows": naive_rows},
        "hybrid": {"aggregate": hybrid_agg, "rows": hybrid_rows},
    }, indent=2))
    print(f"\nFull report written to {out_path}")

    if args.gate:
        passed = apply_gate(hybrid_agg, dry_run=args.dry_run)
        print("\nGATE:", "PASS" if passed else "FAIL")
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()

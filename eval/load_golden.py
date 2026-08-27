"""
Load and VALIDATE the golden eval set against the ingested chunks.

Why this exists: the golden set is your answer key. A single typo'd gold_chunk_id
silently corrupts context_recall — you'd measure retrieval against a chunk that
doesn't exist and never know. This validator makes that failure loud, and it
doubles as a CI gate later (exits non-zero on any problem).

Usage:
    python -m eval.load_golden \
        --golden data/golden/golden_set.jsonl \
        --chunks data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = {"id", "question", "answer", "gold_chunk_ids", "type"}
VALID_TYPES = {"single_hop", "multi_hop", "unanswerable"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(golden: list[dict], chunk_ids: set[str]) -> list[str]:
    """Return a list of problems. Empty list == valid."""
    problems: list[str] = []
    seen_ids: set[str] = set()

    for i, rec in enumerate(golden):
        rid = rec.get("id", f"<row {i}>")

        missing = REQUIRED - rec.keys()
        if missing:
            problems.append(f"{rid}: missing fields {sorted(missing)}")
            continue

        if rec["id"] in seen_ids:
            problems.append(f"{rid}: duplicate id")
        seen_ids.add(rec["id"])

        if rec["type"] not in VALID_TYPES:
            problems.append(f"{rid}: invalid type '{rec['type']}' (allowed: {sorted(VALID_TYPES)})")

        gold = rec["gold_chunk_ids"]
        if not isinstance(gold, list):
            problems.append(f"{rid}: gold_chunk_ids must be a list")
            continue

        if rec["type"] == "unanswerable":
            if gold:
                problems.append(f"{rid}: unanswerable questions must have empty gold_chunk_ids")
        else:
            if not gold:
                problems.append(f"{rid}: answerable question has no gold_chunk_ids")
            # the core check: every referenced chunk must actually exist
            for cid in gold:
                if cid not in chunk_ids:
                    problems.append(f"{rid}: gold_chunk_id does not exist in chunks -> {cid}")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden_set.jsonl")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    args = ap.parse_args()

    golden = load_jsonl(Path(args.golden))
    chunk_ids = {json.loads(l)["chunk_id"] for l in Path(args.chunks).read_text().splitlines() if l.strip()}

    problems = validate(golden, chunk_ids)

    # summary
    by_type: dict[str, int] = {}
    for r in golden:
        by_type[r.get("type", "?")] = by_type.get(r.get("type", "?"), 0) + 1
    print(f"Golden set: {len(golden)} questions across {len(chunk_ids)} chunks")
    for t, n in sorted(by_type.items()):
        print(f"  {t:12s} {n}")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print("\nOK — every gold_chunk_id resolves, schema is clean.")


if __name__ == "__main__":
    main()

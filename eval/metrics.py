"""
metrics.py — scoring functions for eval/run_eval.py.

Two families, matching the project's core insight (a system can score well on
generation metrics while silently failing on retrieval — track both, separately,
or a recall regression hides behind a healthy-looking faithfulness number):

RETRIEVAL-STAGE (deterministic — exact chunk_id set overlap against the golden
set's gold_chunk_ids. No LLM judge, no randomness, fully reproducible run to run):
    context_recall(retrieved_ids, gold_ids)     -> fraction of NEEDED chunks retrieved
    context_precision(retrieved_ids, gold_ids)  -> fraction of RETRIEVED chunks that were relevant

GENERATION-STAGE:
    citation_gold_overlap(cited_ids, gold_ids)  -> deterministic, stricter than recall:
                                                     fraction of gold chunks the model actually
                                                     CITED, not just retrieved. Retrieving the right
                                                     chunk means nothing if generation ignores it.
    judge_correctness(...)                       -> LLM-judged (cheap, gpt-5-mini): does the answer
                                                     match the golden answer's factual content?
    abstention_correct(...)                      -> deterministic: correct abstain/answer behavior.

Both gold-chunk-based metrics return None (not a number) when gold_chunk_ids is
empty — i.e. for the unanswerable bucket, where "did we retrieve the right chunk"
isn't a meaningful question. Callers must skip Nones when averaging, not treat
them as zero — see aggregate() in run_eval.py.
"""
from __future__ import annotations

ABSTAIN_PREFIX = "NOT_ANSWERABLE:"

JUDGE_MODEL = "gpt-5-mini"

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker comparing a candidate answer \
against a known-correct reference answer to the same question.

Judge the candidate CORRECT only if it states the same key fact(s) as the reference \
(same numbers, same entities, same conclusion) — wording may differ, but the substance \
must match. Judge it INCORRECT if it states a different number, attributes a fact to the \
wrong entity, misses the key point, or contradicts the reference.

Reply with EXACTLY one word on the first line — CORRECT or INCORRECT — optionally \
followed by a one-sentence reason on a second line."""


def context_recall(retrieved_ids: set[str], gold_ids: set[str]) -> float | None:
    """Of the chunks actually needed to answer, what fraction did retrieval find?
    This is the PRIMARY gate metric — the one most likely to catch a real regression
    that a generation-only dashboard would miss entirely."""
    if not gold_ids:
        return None
    return len(retrieved_ids & gold_ids) / len(gold_ids)


def context_precision(retrieved_ids: set[str], gold_ids: set[str]) -> float | None:
    """Of the chunks retrieval actually returned, what fraction were relevant?
    Low precision with high recall means retrieval works but is noisy — the
    generator has to sift signal from clutter, which is a real (if lesser) risk."""
    if not gold_ids or not retrieved_ids:
        return None
    return len(retrieved_ids & gold_ids) / len(retrieved_ids)


def citation_gold_overlap(cited_ids: set[str], gold_ids: set[str]) -> float | None:
    """Of the chunks actually needed, what fraction did the model CITE (not just
    retrieve)? Stricter than context_recall — a chunk sitting unused in the
    retrieved set doesn't help if the model never draws on it."""
    if not gold_ids:
        return None
    return len(cited_ids & gold_ids) / len(gold_ids)


def abstention_correct(is_answerable: bool, abstained: bool) -> bool:
    """Correct behavior is binary and unambiguous: abstain iff the question is
    genuinely unanswerable from this corpus. Answering a truly unanswerable question
    (hallucination) and refusing an answerable one (over-caution) are both failures."""
    return abstained != is_answerable  # abstained should be True exactly when NOT answerable


def judge_correctness(question: str, golden_answer: str, candidate_answer: str,
                       model: str = JUDGE_MODEL) -> tuple[bool | None, str]:
    """Real API call (OpenAI). Returns (is_correct, raw_judge_text).
    Kept deliberately separate from run_eval's orchestration so it can be swapped
    or mocked independently — same separation-of-concerns as generate.py's
    build/parse split."""
    from openai import OpenAI
    client = OpenAI()

    prompt = (
        f"Question: {question}\n\n"
        f"Reference (known correct) answer: {golden_answer}\n\n"
        f"Candidate answer: {candidate_answer}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    first_line = text.splitlines()[0].strip().upper() if text else ""
    # "INCORRECT".startswith("CORRECT") is False (different first letters: I vs C) —
    # verified with a direct unit test in test_metrics.py rather than trusted on sight.
    is_correct = first_line.startswith("CORRECT")
    return is_correct, text


def _dry_run_judge_correctness(*args, **kwargs) -> tuple[None, str]:
    """No network call — used when --dry-run is set or no API key is present."""
    return None, "[DRY RUN — judge not called]"

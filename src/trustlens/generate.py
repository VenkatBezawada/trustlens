"""
generate.py — grounded generation with citation enforcement, two swappable backends.

DEFAULT BACKEND: OpenAI (gpt-5-mini). Chosen here purely because that's the key
available for this project — see the "why not Claude" note below for the honest
tradeoff. Anthropic is kept as a second backend (--backend anthropic) since its
citations API gives structurally-verified citations rather than prompted ones,
and comparing the two is a legitimate Phase 2 extension.

--- How citation enforcement works per backend ---

ANTHROPIC: retrieved chunks are passed as "document" content blocks with citations
enabled. The API structurally maps each citation in the response to an exact
document index + text span — the model cannot fake a citation, the platform
verifies the link. See build_documents_anthropic() / parse_anthropic_response().

OPENAI: OpenAI's chat completions API has no equivalent native "pass documents,
get back structurally verified citations" feature — its own citation guidance is a
prompt-formatted convention (you define a marker format, instruct the model to use
it, then resolve it yourself). So here: retrieved chunks are numbered sources in
the prompt; the model is instructed to append a bracket marker like [2] right after
any claim drawn from source 2; we then regex-parse those markers out of the answer
and resolve them back to chunk_ids via the numbered-source list. This is WEAKER
than the Anthropic path — citations are only as reliable as the model choosing to
follow the marker instruction, nothing structurally forces it. Flagging that
honestly rather than presenting it as equivalent.

Both backends produce the SAME GenerationResult shape, so eval/run_eval.py can
score either one identically.

Abstention protocol (shared): the model must reply with the exact prefix
"NOT_ANSWERABLE:" when the provided documents don't contain the answer. A hard,
parseable signal instead of fuzzy string-matching "I don't know" phrasings.

Requires an API key in the environment: OPENAI_API_KEY (default backend) or
ANTHROPIC_API_KEY (--backend anthropic). Use --dry-run to exercise the request-
building and response-parsing code without hitting either API.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m trustlens.generate --query "..." --mode hybrid --k 6
    python -m trustlens.generate --query "..." --backend anthropic --k 6
    python -m trustlens.generate --query "..." --dry-run              # no key needed
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field

# Current model strings as of this build. If either errors, check the provider's
# docs for the current name — model lineups move fast on both sides.
OPENAI_MODEL = "gpt-5-mini"
ANTHROPIC_MODEL = "claude-sonnet-5"

ABSTAIN_PREFIX = "NOT_ANSWERABLE:"

BASE_RULES = f"""Rules:
- Every factual claim in your answer must be grounded in the provided sources.
- If the sources do not contain enough information to answer, reply with EXACTLY \
this prefix followed by a one-sentence reason: "{ABSTAIN_PREFIX} <reason>"
- Do not guess, estimate, or fill gaps with plausible-sounding numbers.
- Be concise — answer the question directly, do not restate it."""


@dataclass
class Citation:
    chunk_id: str
    cited_text: str


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    cited_chunk_ids: set = field(default_factory=set)
    abstained: bool = False
    backend: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "cited_chunk_ids": sorted(self.cited_chunk_ids),
            "abstained": self.abstained,
            "backend": self.backend,
        }


# ============================== OpenAI backend ==============================
# Bracket-citation convention: [1], [2], ... map 1-indexed to the numbered
# source list built from the retrieved chunks, in order.

_BRACKET_RE = re.compile(r"\[(\d+)\]")

OPENAI_SYSTEM_PROMPT = f"""You are answering questions using ONLY the numbered sources \
provided below (excerpts from Indian company annual reports). Do not use outside knowledge.

Citation format: immediately after any sentence or clause that uses information from \
a source, append its number in square brackets, e.g. "Revenue grew 10% [2]." If a claim \
draws on multiple sources, cite all of them: "...as seen in both reports [1][3]."

{BASE_RULES}"""


def build_prompt_openai(query: str, chunks: list[dict]) -> str:
    """Numbered-source prompt. Source numbers are 1-indexed and positionally
    match `chunks`, in order — that's how a [N] marker resolves back to a chunk_id."""
    lines = ["Sources:\n"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({c['company']}, {c['fy']}, p.{c['page']}): {c['text']}\n")
    lines.append(f"\nQuestion: {query}")
    return "\n".join(lines)


def parse_openai_response(answer_text: str, chunks: list[dict]) -> GenerationResult:
    """Pure parsing function — deliberately separate from the API call so it can be
    unit-tested with a plain string, no network or mock objects required."""
    answer_text = answer_text.strip()
    abstained = answer_text.startswith(ABSTAIN_PREFIX)

    citations: list[Citation] = []
    cited_chunk_ids: set[str] = set()

    if not abstained:
        # walk sentence-ish spans so each citation can carry the text it supports
        for sent in re.split(r"(?<=[.!?])\s+", answer_text):
            for num_str in _BRACKET_RE.findall(sent):
                idx = int(num_str) - 1  # convert 1-indexed marker to 0-indexed chunks position
                if 0 <= idx < len(chunks):
                    cid = chunks[idx]["chunk_id"]
                    clean_sent = _BRACKET_RE.sub("", sent).strip()
                    citations.append(Citation(chunk_id=cid, cited_text=clean_sent))
                    cited_chunk_ids.add(cid)
                # out-of-range marker numbers are silently ignored, not fatal —
                # a hallucinated source number shouldn't crash the pipeline.

    return GenerationResult(
        answer=answer_text, citations=citations,
        cited_chunk_ids=cited_chunk_ids, abstained=abstained, backend="openai",
    )


def generate_openai(query: str, chunks: list[dict], model: str = OPENAI_MODEL) -> GenerationResult:
    """Real API call. Requires OPENAI_API_KEY in the environment."""
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env automatically

    prompt = build_prompt_openai(query, chunks)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer_text = response.choices[0].message.content or ""
    return parse_openai_response(answer_text, chunks)


def _dry_run_openai(query: str, chunks: list[dict]) -> GenerationResult:
    """No network call. Builds plausible completion text with a bracket citation
    on the top chunk, then runs it through the REAL parser — proves the parsing
    path works without spending an API call."""
    if not chunks:
        text = f"{ABSTAIN_PREFIX} no documents were retrieved."
    else:
        text = "[DRY RUN — no real API call made] Placeholder answer citing the top source [1]."
    return parse_openai_response(text, chunks)


# ============================= Anthropic backend =============================
# Structural citations API — see module docstring. Kept for comparison / Phase 2.

ANTHROPIC_SYSTEM_PROMPT = f"""You are answering questions using ONLY the provided documents \
(excerpts from Indian company annual reports). Do not use outside knowledge.

{BASE_RULES}"""


def build_documents_anthropic(chunks: list[dict]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c["text"]},
            "title": c["chunk_id"],
            "citations": {"enabled": True},
        }
        for c in chunks
    ]


def parse_anthropic_response(response, chunks: list[dict]) -> GenerationResult:
    chunk_ids_by_doc_index = [c["chunk_id"] for c in chunks]
    answer_parts: list[str] = []
    citations: list[Citation] = []
    cited_chunk_ids: set[str] = set()

    for block in response.content:
        if block.type != "text":
            continue
        answer_parts.append(block.text)
        for cit in (getattr(block, "citations", None) or []):
            doc_idx = cit.document_index
            if 0 <= doc_idx < len(chunk_ids_by_doc_index):
                cid = chunk_ids_by_doc_index[doc_idx]
                citations.append(Citation(chunk_id=cid, cited_text=cit.cited_text))
                cited_chunk_ids.add(cid)

    full_answer = "".join(answer_parts).strip()
    return GenerationResult(
        answer=full_answer, citations=citations, cited_chunk_ids=cited_chunk_ids,
        abstained=full_answer.startswith(ABSTAIN_PREFIX), backend="anthropic",
    )


def generate_anthropic(query: str, chunks: list[dict], model: str = ANTHROPIC_MODEL) -> GenerationResult:
    import anthropic
    client = anthropic.Anthropic()
    documents = build_documents_anthropic(chunks)
    response = client.messages.create(
        model=model, max_tokens=1024, system=ANTHROPIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": documents + [{"type": "text", "text": query}]}],
    )
    return parse_anthropic_response(response, chunks)


class _FakeCitation:
    def __init__(self, document_index, cited_text):
        self.document_index = document_index
        self.cited_text = cited_text


class _FakeTextBlock:
    def __init__(self, text, citations=None):
        self.type = "text"
        self.text = text
        self.citations = citations


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def _dry_run_anthropic(query: str, chunks: list[dict]) -> GenerationResult:
    if not chunks:
        fake = _FakeResponse([_FakeTextBlock(f"{ABSTAIN_PREFIX} no documents were retrieved.")])
    else:
        fake = _FakeResponse([_FakeTextBlock(
            "[DRY RUN — no real API call made]",
            citations=[_FakeCitation(document_index=0, cited_text=chunks[0]["text"][:80])],
        )])
    return parse_anthropic_response(fake, chunks)


# ================================= dispatch ==================================

def generate(query: str, chunks: list[dict], backend: str = "openai", dry_run: bool = False) -> GenerationResult:
    key_present = os.environ.get("OPENAI_API_KEY" if backend == "openai" else "ANTHROPIC_API_KEY")
    if dry_run or not key_present:
        return _dry_run_openai(query, chunks) if backend == "openai" else _dry_run_anthropic(query, chunks)
    return generate_openai(query, chunks) if backend == "openai" else generate_anthropic(query, chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--mode", choices=["naive", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chroma-dir", default=".chroma")
    ap.add_argument("--bm25-path", default="data/processed/bm25.pkl")
    args = ap.parse_args()

    from trustlens.retrieve import Retriever
    retriever = Retriever(chroma_dir=args.chroma_dir, bm25_path=args.bm25_path)
    chunks = retriever.retrieve(args.query, mode=args.mode, k=args.k)
    print(f"Retrieved {len(chunks)} chunks ({args.mode} mode): {[c['chunk_id'] for c in chunks]}")

    key_env = "OPENAI_API_KEY" if args.backend == "openai" else "ANTHROPIC_API_KEY"
    if not args.dry_run and not os.environ.get(key_env):
        print(f"No {key_env} set — falling back to --dry-run.")

    result = generate(args.query, chunks, backend=args.backend, dry_run=args.dry_run)

    print(f"\nQuery: {args.query}")
    print(f"Backend: {result.backend}   Abstained: {result.abstained}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nCited chunk_ids: {sorted(result.cited_chunk_ids)}  ({len(result.citations)} citation span(s))")


if __name__ == "__main__":
    main()

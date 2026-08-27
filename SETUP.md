# Local Setup

*Everything so far was built inside Claude's temporary sandbox — this gets it running on your own machine, which is where it needs to live for a real portfolio project (and where you'll eventually push it to GitHub).*

Pick your platform below — **macOS/Linux (bash/zsh)** or **Windows (PowerShell)**. The two use different syntax for almost every step (env vars, `&&`, venv activation), so don't mix commands from the wrong section.

---

## macOS / Linux (bash or zsh)

**1. Unzip and enter the project**
```bash
unzip trustlens.zip
cd trustlens
```

**2. Python environment** (requires Python 3.11+)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -e .
```

**4. Set your OpenAI API key** (get one at platform.openai.com → API Keys). The project defaults to the OpenAI backend — pass `--backend anthropic` to any command instead if you have an Anthropic key and want the structurally-verified citations path (see `generate.py`'s module docstring for the tradeoff).
```bash
export OPENAI_API_KEY=sk-...
```
Lasts for this terminal session only. To persist it, add that line to `~/.zshrc` or `~/.bashrc` — but never commit it or paste it anywhere shared.

**5. Verify it works**
```bash
PYTHONPATH=src python -m trustlens.retrieve --query "What was India's real GDP growth in FY24?" --mode hybrid --k 4
PYTHONPATH=src python -m trustlens.generate --query "What was Cipla's R&D spend growth in FY24?" --mode hybrid --k 4
```

---

## Windows (PowerShell)

**1. Unzip and enter the project**
```powershell
Expand-Archive trustlens.zip -DestinationPath .
cd trustlens
```
(`unzip` and `&&` are bash syntax — PowerShell doesn't recognize either. Use `Expand-Archive` and run commands on separate lines, or join with `;` instead of `&&`.)

**2. Python environment** (requires Python 3.11+)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If activation errors with "running scripts is disabled on this system," run this once, then retry:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
(If `python` isn't found, try `py` instead — some Windows installs only register that alias.)

**3. Install dependencies**
```powershell
pip install -e .
```

**4. Set your OpenAI API key** (get one at platform.openai.com → API Keys). The project defaults to the OpenAI backend — pass `--backend anthropic` to any command instead if you have an Anthropic key and want the structurally-verified citations path (see `generate.py`'s module docstring for the tradeoff).
```powershell
$env:OPENAI_API_KEY = "sk-..."
```
`export` is bash-only — PowerShell uses `$env:VAR = "value"`. This lasts for the current window only; open a new PowerShell window and you'll need to set it again (or add it to your PowerShell `$PROFILE` script to persist it).

**5. Verify it works**

PowerShell doesn't support inline `VAR=value command` syntax — set the env var on its own line first:
```powershell
$env:PYTHONPATH = "src"
python -m trustlens.retrieve --query "What was India's real GDP growth in FY24?" --mode hybrid --k 4
python -m trustlens.generate --query "What was Cipla's R&D spend growth in FY24?" --mode hybrid --k 4
```

---

## Common to both platforms

**Never paste a real API key into a chat with me or anyone else** — chat is a shared, logged channel. Keep it local only (env var or `.env` file, never committed to git).

**The vector index is already built.** This zip includes a working `.chroma/` and `data/processed/bm25.pkl`, verified before packaging — you don't need to rebuild anything to run step 5 above. Only rebuild (`python -m trustlens.index`, with `PYTHONPATH`/`$env:PYTHONPATH` set to `src`) if you change the chunking or add new companies.

**(Optional) Use a `.env` file instead of setting the key each session:**
```bash
pip install python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env    # already in .gitignore
```
Then in any script that calls the API:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Retrieval needs no key at all** — only generation does. If the relevant key (`OPENAI_API_KEY` by default, or `ANTHROPIC_API_KEY` with `--backend anthropic`) isn't set, `generate.py` automatically falls back to `--dry-run` and tells you, rather than failing silently.

**Raw PDFs aren't included** (~50MB, unnecessary — extracted text is already in `chunks.jsonl`). Source URLs are in `data/manifest.json` under each entry's `source_url` if you want them.

---

**Where you are in the project:** ingest ✓, golden set ✓ (16 questions), retrieval ✓ (naive + hybrid), generation ✓ (citation-enforced, tested in dry-run — live calls need your key). **Next:** `eval/run_eval.py` — runs both retrieval modes against the golden set and produces the before/after numbers for the README.

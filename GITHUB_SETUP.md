# GitHub + CI Setup

*One-time setup to get this project on GitHub with the eval gate actually running. Git commands are identical in PowerShell and bash — no platform differences here.*

## 1. Check you have git
```powershell
git --version
```
If that errors, install from git-scm.com, or use GitHub Desktop (a GUI alternative) instead of the commands below.

## 2. Initialize the repo locally
From inside the `trustlens` folder:
```powershell
git init
git add .
git commit -m "Initial commit: eval-driven RAG over Indian annual reports"
```
Check `git status` first if you're unsure what will be included — `.gitignore` already excludes `.venv/`, `.chroma/`, `data/raw/*.pdf`, `data/processed/bm25.pkl`, and `.env`. Everything else (code, `chunks.jsonl`, the golden set, the workflow file) gets tracked.

## 3. Create the GitHub repo
Go to **github.com/new**, create a repo (e.g. `trustlens`), **don't** initialize it with a README (you already have files locally — that would conflict).

Then connect and push:
```powershell
git remote add origin https://github.com/<your-username>/trustlens.git
git branch -M main
git push -u origin main
```

## 4. (Optional) Add your OpenAI key for the real gate
**You don't need to do this at all.** The workflow now detects whether a key is configured and adapts automatically:
- **No key set** → CI runs `eval/run_eval.py --dry-run` — free, no API calls, no key needed anywhere. This validates the whole pipeline (ingest, index, retrieve) runs correctly and reports real, deterministic retrieval metrics. It does not enforce the quality gate, since correctness can't be judged for free.
- **Key set** → CI runs the real, paid gate — actual generation, actual LLM-judged correctness, and a genuine pass/fail against the thresholds in `eval/run_eval.py`.

If you want the real gate later, add it as a **repo secret** (not in any file, never committed): your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**, name `OPENAI_API_KEY`. GitHub encrypts secrets and never exposes them in logs or code — this is a different thing from putting a key in a file, which is what `.gitignore`/`.env` already protect against. Add it whenever you're comfortable; the same workflow picks it up automatically, no code changes needed.

## 5. Watch it run
Your push in step 3 already triggered the workflow once. Check your repo's **Actions** tab. Without a key configured, expect a **green check** — that's the dry-run pipeline-validation path succeeding, not a quality gate passing (that distinction matters if you're describing this in an interview).

## 6. The portfolio moment: demonstrate the gate actually blocking something
This is the screenshot worth having in your README — proof the gate isn't decorative.

```powershell
git checkout -b break-retrieval
```
Make a deliberately bad change — e.g. in `src/trustlens/retrieve.py`, change hybrid mode's `OVER_RETRIEVE_N` from 15 down to 1 (starves the reranker of candidates), or in `eval/run_eval.py` temporarily force `mode="naive"` regardless of the loop. Commit and push:
```powershell
git add .
git commit -m "Deliberately break retrieval to demo the CI gate"
git push -u origin break-retrieval
```
Open a Pull Request on GitHub from this branch into `main`. Watch the **RAG Eval Gate** check run and fail (red ❌) directly on the PR. Screenshot that — it's the single most convincing image for a README, because it proves the gate can't be bypassed by a bad change slipping through unnoticed.

Then close the PR without merging, and delete the branch:
```powershell
git checkout main
git branch -D break-retrieval
git push origin --delete break-retrieval
```

## 7. Cost note
Each real gate run makes ~32 generation calls + up to 32 judge calls against `gpt-5-mini` — a few cents per run (see the cost breakdown discussed earlier in the project). Fine to trigger a handful of times while setting this up; not something to run on every commit indefinitely without noticing.

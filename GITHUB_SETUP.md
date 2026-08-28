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

## 6. The portfolio moment: get a real, honest red ❌ on a PR

Good news: you don't need to fake a regression. `context_recall` is a genuine, deterministic metric — no API key needed for it — and your real hybrid recall (77.4%) is honestly below the gate's 80% threshold right now. The gate was recently changed to actually *enforce* that free check even with no key configured (it used to just report the number and skip enforcement). Turning that enforcement on is itself a real, defensible PR — and it will genuinely fail, for a genuine, already-documented reason.

```powershell
git checkout -b enforce-recall-gate
git add .
git commit -m "Enforce free recall-only gate even without an API key"
git push -u origin enforce-recall-gate
```

Open a Pull Request on GitHub from this branch into `main`. Watch the **RAG Eval Gate** check run and fail (red ❌) directly on the PR — a real quality bar, genuinely not met yet, caught automatically, no artificial sabotage and no API key required. Screenshot that.

**Whether to merge it is a real decision, not just cleanup:** merging means your `main` branch's CI badge turns red too, honestly reflecting that neither retrieval mode clears 80% recall yet — which is already exactly what the README says in plain text. An honest red badge with a documented, explained cause is more credible than a misleadingly green one. If you'd rather keep the badge green for someone quickly scanning the repo, you can leave the PR open (or close it) without merging — the screenshot alone still proves the mechanism works. Either choice is defensible; know which story you're telling.

### Optional: the artificial-sabotage variant
If you later improve recall past 80% (bigger golden set, better chunking) and want to *re-*demonstrate the gate catching a *new* regression, the original recipe still works: branch off, deliberately break something (e.g. in `src/trustlens/retrieve.py`, drop hybrid's `OVER_RETRIEVE_N` from 15 to 1), open a PR, watch it fail, then revert without merging.

## 7. Cost note
Each real gate run makes ~32 generation calls + up to 32 judge calls against `gpt-5-mini` — a few cents per run (see the cost breakdown discussed earlier in the project). Fine to trigger a handful of times while setting this up; not something to run on every commit indefinitely without noticing.

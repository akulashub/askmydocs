# Getting the project live on GitHub

Seven steps, about an hour including waiting for CI. Steps 1–3 need nothing but a GitHub account.

## 0. What you need

- A GitHub account and `git` installed (`git --version`).
- Python 3.10+ (only for the optional local check in step 1).
- For steps 4–6: an Anthropic API key from https://console.anthropic.com (Settings → API keys).

## 1. Unzip and check it works locally (optional, 2 minutes)

```bash
unzip askmydocs.zip && cd askmydocs
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,api,loaders]"
pytest -q                                               # expect: 25 passed
askmydocs --profile offline eval                        # expect: PASS ✅
```

## 2. Put it on GitHub

1. On github.com click **New repository**. Name it (e.g. `askmydocs`), make it **Public** (it's a portfolio
   piece), and **don't** tick "Add a README" — the repo must start empty.
2. The zip has no git history, so initialise it and push:

```bash
git init
git add -A
git commit -m "Ask My Docs: citation-enforced RAG with CI-gated evaluation"
git branch -M main
git remote add origin https://github.com/<your-username>/askmydocs.git
git push -u origin main
```

If the push asks for a password, GitHub wants a personal access token, not your account password:
github.com → Settings → Developer settings → Personal access tokens. Or install the GitHub CLI and run
`gh auth login` once.

## 3. Watch the first CI run

Open the repo's **Actions** tab. The `ci` workflow starts automatically. What to expect:

| job | first run | why |
|---|---|---|
| tests + offline eval | ✅ green | baseline already committed |
| eval (neural retrieval, no API key) | ✅ green, with a ⚠️ "no baseline yet" note | downloads bge + cross-encoder (~5 min first time, cached after) |
| eval (Claude + RAGAS) | skipped | no API key yet |

Click a job → **Summary** to see the metrics table. If the offline job is red, open its summary and send me
the "Gate failures" section.

## 4. Add your Anthropic API key as a secret

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

- Name: `ANTHROPIC_API_KEY`
- Secret: your key (starts with `sk-ant-`)

GitHub never shows it in logs. Each Claude eval run makes a few hundred API requests (77 answers plus the
RAGAS judge's calls), so check usage at console.anthropic.com after the first run.

## 5. Run calibration (one click, ~20–40 minutes)

**Actions** → **calibrate** (left sidebar) → **Run workflow** → keep the default `models default` → **Run**.

It runs the real-model eval and the full Claude eval, writes a baseline for each, and suggests thresholds.
When it finishes, open the run and download the **calibration** artifact (bottom of the summary page).

## 6. Review and commit the calibration

Inside the artifact:

- `baselines/models.json`, `baselines/default.json` → copy into `eval/baselines/` in your repo.
- `baselines/SUGGESTED_THRESHOLDS.md` → replace the `models:` and `default:` blocks in
  `eval/thresholds.yaml` with the suggested ones.
- `reports/latest-*.md` → **read these before committing**. A baseline locks in whatever quality you have
  now, so look at the "Items to inspect" table. A few misses are normal; if, say, refusal accuracy is
  very low, fix that first (or tell me what you see).

```bash
git add eval/baselines eval/thresholds.yaml
git commit -m "Calibrate model and Claude eval baselines"
git push
```

From this push on, all three tiers enforce their gates.

## 7. Prove the gate works (great for the portfolio)

```bash
git checkout -b demo/break-retrieval
# in config/settings.yaml, set   rerank_top_n: 1
git commit -am "Demo: starve the LLM of context"
git push -u origin demo/break-retrieval
```

Open a pull request. CI should go red. This is the actual offline-tier output for that change:

```
EVAL GATE FAILED:
  context_recall@k=0.739 below threshold 0.75
  context_recall@k regressed 0.985 -> 0.739 (tolerance 0.03)
  mrr@k regressed 0.842 -> 0.739 (tolerance 0.03)
  citation_doc_accuracy regressed 0.897 -> 0.860 (tolerance 0.03)
```

Screenshot the red check for your README, then close the PR without merging.

## Tips

- **Cost control:** the Claude tier runs on every PR. To run it only on merges to `main`, add
  `if: github.event_name != 'pull_request'` under the `llm:` job in `.github/workflows/ci.yml`.
- **PRs from forks** don't get secrets, so the Claude tier is skipped for them. That's expected.
- **`judge_coverage` failure** means the RAGAS judge errored (bad key, rate limit). Re-run the job.
- **Prefer running locally?** `pip install -e ".[models,anthropic,ragas]"`, export the key, then
  `make baseline PROFILE=models`, `make baseline PROFILE=default`, and
  `python scripts/suggest_thresholds.py models default`.
- **Re-calibrate** whenever you deliberately change quality (new corpus, new model): run step 5 again.

# Handoff: RSA loop smoke test 2 (Gemini 3.8 Flash, all data, two rounds)

**Goal.** The last check before Sherlock. Smoke test 1 (`SMOKE_RESULTS.md`)
showed agents can get memo models admitted, but these were never exercised:

- a self-check that finishes and that agents read;
- a rejection followed by a repair;
- Gemini 3.8 Flash (now the RSA default);
- the external data download;
- the larger combined dataset;
- the new pruning unit.

This run exercises all of them.

**Where.** A fresh claude.ai cloud session on branch `auto-rsa`, in a
disposable container, so `--no-sandbox` is acceptable here and nowhere else.
`GOOGLE_API_KEY` must be set (check with `[ -n "$GOOGLE_API_KEY" ] && echo set`;
never print it).

## Steps

```bash
git checkout auto-rsa && git pull && uv sync
npm i -g opencode-ai@1.18.35
export GOOGLE_GENERATIVE_AI_API_KEY="$GOOGLE_API_KEY"
cd /tmp && opencode run -m google/gemini-3.8-flash "Reply with the word ok." ; cd -

# 1. The data download (another failure mode on a fresh machine). It fetches
#    the three sources without a licence from their pinned URLs, checks each
#    sha256 and rebuilds the CSVs.
uv run python -m src.rsa.ingest.run --sources mayn_demberg_2026 mayn_demberg_2023 mayn_demberg_2022 sikos_2021
RSA_INGEST_FETCH=1 uv run pytest -q tests/test_rsa_ingest_*.py   # must pass; byte-identical rebuilds
uv run python -m src.rsa.ingest.combine \
    --sources pragmods mayn_demberg_2026 mayn_demberg_2023 mayn_demberg_2022 sikos_2021 \
    --out data/rsa/combined_trials.csv

# 2. The loop: 2 rounds x 3 slots on all five sources, with the RSA defaults
#    (google/gemini-3.8-flash, 2,400 s per agent). Expect a few hours: the
#    seed fits on about 50k forced-choice trials are slower than on pragmods.
uv run python -m src.rsa.loop.run --results data/rsa/loop_smoke2 \
    --responses data/rsa/combined_trials.csv \
    --max-iterations 2 --candidate-count 3 --no-sandbox \
    --num-warmup 500 --num-samples 500 --num-chains 2
```

If the seed fits on the combined data take over about 45 minutes, record the
time per model (that number sizes the Sherlock job), then fall back to
pragmods plus `mayn_demberg_2026` for the agent rounds and say so.

## What to report (in `docs/auto_rsa/SMOKE_RESULTS_2.md`)

- **Download.** Did every source fetch and rebuild byte-identically? How long
  did it take?
- **Fit times.** Per seed model on the combined data (this sizes Sherlock
  jobs), and per admission fit.
- **Per slot.**
  - Outcome and the exact rejection reason.
  - Whether the self-check finished and the agent read its PASS/FAIL line.
  - Any repair, and whether it succeeded.
  - Memo errors hit.
  - Tokens and wall time (from `token_usage.jsonl`).
  - Gemini 3.8 Flash latency (first tool call).
- **Round 2.** Does it see round 1's admitted models in its briefs and build
  on them? Any convergence of the slots on one idea?
- **The end-of-run prune.** What was pruned, and at what clustered-SE ratio?
- **Disk.**
  - Confirm no `.xdg_data/opencode/snapshot` directories (snapshots are now
    off in `opencode.json`).
  - Report the run directory's total size.
- **Bugs.** Any bug that blocks the run: fix it with a test if it is small and
  clear, and describe it.

Commit the results document, any fix, and the small outputs to branch
`auto-rsa-smoke2`. Small outputs means report.html, report.bundle.json,
history.json, attempted_hypotheses.jsonl, token usage, and each candidate's
hypothesis.md, model_name.txt and candidate.py. Leave out .fit_cache,
responses.csv and the external cache. Do not push to `auto-rsa` or `main`,
and do not open a pull request.

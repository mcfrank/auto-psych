# Cluster & live runs

Moved verbatim from `CLAUDE.md` (2026-10-28); `CLAUDE.md` keeps the rules that span files.


- **Clusters:** Sherlock by default; `CLUSTER=coco` runs on `COCO_NODE`
  (default `cocoflops1`; `/scr` is node-local, so a chain runs on one node).
  Never run heavy work on a login node; keep job I/O on scratch. Playwright
  browser simulation does not run on Sherlock compute nodes — use
  `--mode simulated_participants_nobrowser`. Runbooks:
  `scripts/subjective_randomness/README.md` and `slurm/README.md`.
- **Sweeps** (`submit_holdout_test_retest.sh`): setup → array → retry →
  summary. The setup job stages the code once (`harness_repo`, `agent_src`,
  `code_commit`); a sweep runs on **one code**, and a cell refuses to resume on
  other code. The retry job resubmits timeouts, crashes and out-of-memory cells
  (up to 2 rounds); `MISSING_CELLS.txt` lists cells still without a result.
  The impossible-model sweep (the literature sweep's control) is the same chain
  with its own config, ground truths and `GT_MODELS_REL`; its config must equal
  the faithful one except `gt_models`/`gt_models_dir`
  (`tests/test_impossible_config_matches_faithful.py`).
- **Live runs recruit real people and spend real money**: double-gated
  (`confirm_live_recruitment: true` in the config **and**
  `--confirm-live-recruitment`; launchers require typing `yes`).
  `scancel` does **not** stop a published Prolific study — use the dashboard.
  Runbook: `scripts/outer_loop_live/README.md`.
- **Live-study rules** (each enforced in code):
  - One participant count: `--n-participants` sets places, design N and the
    collection target; the rendered `prolific_config.yaml` carries none and is
    never committed.
  - **One study per experiment**: an experiment whose deployment manifest
    records a study refuses to redesign, reimplement or redeploy
    (`LiveStudyAlreadyRecorded`); recover with
    `RESUME_AGENTS=4_collect:5_model_loop`, or deliberately with
    `--publish-another-prolific-study`. `--resume` skips no stage, and
    `start_full_run.sh RUNS=<i>` deletes run *i*'s results, so a running run is
    moved onto fixed code by hand (runbook § 9).
  - A collection that gives up after its 3-hour poll (`_PROLIFIC_MAX_WAIT_SEC`,
    user decision 2026-09-28) **pauses** its study (reversible) and models the
    partial data.
  - Every study excludes earlier participants (`previous_studies_blocklist`
    over the pipeline's published studies); IDs are never written to a config.
  - The page and functions are deployed and verified before the draft study
    is created, so a failed deploy records no study.
  - Each series of parallel runs deploys to its own Hosting sites
    (`<firebase_project>-<run_label>-run<i>`; with `auto-psych-2c5da`,
    `run_label` has 8 characters). All runs share `functions/` (`/submit`,
    `/results`, and RSA's `/assign`).
  - Launchers run from rsync copies without `.git` and record
    `code_provenance.json`; rsyncing fixed code deletes it — record it again.
  - Participant ids are unique across a run's experiments (a returning person
    keeps their id).
  - Jobs default to the `mcfrank` partition (up to 7 days); `--qos=long` is
    not on this account.
- **Secrets** live in repo-root `.secrets` (`.secrets.example`):
  `PROLIFIC_API_TOKEN`, `FIREBASE_TOKEN`, `AUTO_PSYCH_RESULTS_TOKEN` (guards
  `/results`), `GOOGLE_API_KEY` (opencode/Gemini agents), and for Claude
  agents `CLAUDE_CODE_OAUTH_TOKEN` and/or `ANTHROPIC_API_KEY`.
- **Claude billing is stated per run, never guessed**: the `claude` backend
  needs `CLAUDE_AUTH` / `--claude-auth` (`subscription`, `api`, or
  `subscription_then_api`: the API key from a subscription limit until its
  stated reset) and that credential, checked before any agent starts. Each
  agent gets only the credential in use (the CLI prefers an API key over the
  OAuth token). The mode is recorded in token usage and `holdout.json`.

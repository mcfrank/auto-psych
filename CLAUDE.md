# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An automated cognitive-science discovery pipeline. Coding agents conjecture
models of human judgment, design maximally informative experiments, run them on
real (Prolific + Firebase) or simulated participants, and let Bayesian model
comparison decide which hypothesis survives. Three domains: *subjective
randomness* and the *number game* share the PyMC pipeline (everything below
until **RSA reference games**), and *RSA reference games* (memo models fitted
with numpyro, `src/rsa/`, a separate stack). `README.md` is the user-facing
guide; this file captures the architecture and the rules that span files.

## Environment & commands

`uv` drives everything; a plain `uv sync` installs the `dev`, `pymc` and `rsa`
groups (`open-models`, torch/transformers, is opt-in).

```bash
uv sync
uv run pytest -q                        # full suite incl. real MCMC (slow marker = runs NUTS)
uv run pytest -q -m "not slow"          # fast suite
uv run pytest tests/test_eig_selection.py::test_name
./scripts/test / ./scripts/test_fast    # wrappers
```

- The full suite needs more than 5 GB of memory (the JAX/numpyro RSA tests); on
  a small interactive allocation it dies silently partway.
- CI (`.github/workflows/tests.yml`): `uv sync --locked`,
  `tests/test_python_sources_compile.py`, then the fast suite. No committed
  linter/formatter config; match surrounding style.
- **Python 3.12** (memo needs ≥ 3.12). **Keep the `arviz<1` / `pymc<6` caps**
  (newer releases rename the API the model comparison uses) and the
  jax/jaxlib 0.7.0, numpyro 0.19.0 and numpy/scipy/pandas pins: they keep wheels
  resolving on Sherlock's glibc 2.17 nodes.

## Conventions (from the user's global instructions — they apply here)

- Filepaths via `pyprojroot.here()`; CLIs are `tyro` dataclasses (`--help` lists
  every knob), run under `uv run`.
- **Fail loudly.** The codebase deliberately raises on missing models, corrupt
  artifacts, absent credentials, etc. Never add a silent default or fallback.
- BDD dual-loop TDD: a failing integration test first, then red-green-refactor.

## Architecture — the mental model

Two nested loops. **Stages are decoupled processes that pass state through
on-disk artifacts (CSV / JSON / YAML / `.py` files), not in-memory objects.**
Every cognitive model is a self-contained `.py` file listed in a
`models_manifest.yaml`; MCMC fits are content-addressed and cached. The only
state crossing an experiment boundary: the carried model files (the live set),
the ledger of attempted hypotheses beside them, and the registry (a uniform
prior over the carried set).

### Outer loop — `src/pipelines/outer_loop/` (`run.py` → `orchestrator.py`)

`AGENT_KEYS = ["2_design", "3_implement", "4_collect", "5_model_loop"]` (there
is no stage 1). First the model set: experiment 1 seeds from
`projects/<id>/seed_models/`, later experiments carry forward the previous
experiment's set — only once that experiment's model-loop stage validates. Seed
and carry are idempotent and atomic (built beside `cognitive_models/` and
renamed in, `src/runtime/atomic_files.py`), which makes `--resume` and
`--agent <stage>` safe.

- `2_design` — **programmatic.** `eig.design_exhaustive` scores the task's whole
  design pool and greedily selects the max-joint-EIG set (64 by default) →
  `design/stimuli.json`. The EIG counts every participant's response
  (`n_responses` = the participant count, required everywhere). Selection stops
  once the best gain is within Monte Carlo noise of zero; the remaining slots
  are filled by single-response EIG (`source: eig_single_response_fill`).
  Experiment 1 uses prior predictions; later experiments fit on all data so far.
  The search is lazy batched greedy in float32 (an approximation validated by
  `scripts/subjective_randomness/validate_lazy_eig.py`), and each Monte Carlo
  scenario **leaves out the draw that generated it** (`DESIGN_LEAVE_ONE_OUT`),
  so a model needs ≥ 2 draws.
- `3_implement` — the one coding-agent stage: writes a jsPsych experiment
  (skipped in `simulated_participants_nobrowser`); a Firebase/Prolific deploy
  follows when `--deploy-target != none`.
- `4_collect` — programmatic: writes `data/responses.csv`.
- `5_model_loop` — runs the inner loop, exports the live set, writes
  `model_registry.yaml` and last `model_loop/export_complete.json`, which the
  validator checks against `cognitive_models/`, its ledger and the registry.
  Every (re)start goes through `begin_model_loop_stage`: the first records the
  input set (`cognitive_models_input/`) and the run's notes
  (`agent_notes_at_start/`); a restart restores both — notes from the abandoned
  attempt are **discarded** (later agents read the run tree) — and empties
  `model_loop/`. A `model_loop/` with content but no recorded input raises.

Coding stages that fail their validator (`_validate_*` in
`orchestrator_validators.py`) are re-spawned with the error as feedback;
programmatic stages fail terminally.

### Inner loop — `src/pipelines/inner_loop/`

`pymc_orchestrator.py` orchestrates; `model_zoo.py` does seeding, admission,
pruning and the novelty gate; `scoring.py` scoring and export;
`candidate_agent.py` briefs and spawns agents. Each round: optional critique →
candidate agents in parallel → sequential admission.

- **Starting models are not protected** (user decision 2026-09-28): pruned,
  capped and dropped like any model. They are recorded once in
  `<run>/starting_models.json` (`{"starting_models": [...],
  "starting_models_prunable": true}`); a bare-list record means a run started
  with protected seeds and refuses to continue (the conditions must not mix).
  No candidate may reuse a starting model's name or that of a model retired
  earlier in the run (`reserved_names`); clashes are renamed `<name>_2`.
- **Admission** (`_admit_candidate`): a loadable `candidate.py` (module-level
  `model: pm.Model`) plus `hypothesis.md`, passing in order: the **code gate**
  (`import_gate.py`: import allowlist, no file access or reflection escapes —
  agent code runs in the harness process); the logp gate; the **data contract**
  (`src/models/model_contract.py`: the likelihood must be Bernoulli(chose_left;
  p_left) over the CSV's rows, no `pm.Potential` on the responses; a starting
  model that breaks it raises); the fit and finite-ELPD gates; the
  **convergence gate** (≤0.1% divergent, R-hat ≤ 1.05, bulk ESS ≥ 100; a near
  miss is refit once at `target_accept` 0.95; a model's declared
  `target_accept` is a floor, so rejections advise reparameterising); the
  **time limit** (`CANDIDATE_FIT_TIME_LIMIT_SEC` = 30 min per sampling run,
  user decision 2026-09-28; starting and carried models are never limited); and
  the **novelty gate** (posterior-mean `p_left` ≥ `novelty_rmse_threshold`,
  0.002, RMSE from every admitted model on the loop's own 512-stimulus novelty
  pool — deliberately not the training stimuli nor the recovery harness's eval
  pool). A candidate binding `trial_index`, or undefined or failing on the
  pool, is rejected. A code error raised in the candidate's own file is a
  rejection; the same error from the harness, or an infrastructure error,
  raises (`is_model_failure`).
- **Retry and repair** (`_Slot`): a slot whose agent wrote nothing is
  re-spawned once (`candidate_<i>_retry_1/`); a rejected candidate is
  re-spawned once with the rejection verbatim (`candidate_<i>_repair_1/`); a
  repair is final. Every attempt is a ledger line. An agent stopped by a usage
  or rate limit (`src/runtime/usage_limits.py`) waits for the stated reset and
  reruns, up to 12 h (`AgentUsageLimitExceeded`); a login that cannot pay
  raises `AgentLoginFailed`. Agents self-check with `check_candidate.py`.
- **Slot roles** (`slot_roles`): with ≥ 4 slots, `C - 3` exploratory slots walk
  the task's lens battery (`TaskSpec.candidate_hints`), two **refine the
  incumbent** and one **refines a model of the agent's choosing** from
  `refinement_menu.md` (live and pruned models with hypotheses and sources).
  Refinement briefs lift the anti-grafting and no-composition clauses. The role
  is the slot's assignment, recorded in the ledger context; which model an
  agent refined is prose in its `hypothesis.md` and is **never parsed**. The
  holdout configs run five rounds of six per experiment.
- **Pruning** (`_prune_losers`) runs **once, at the end of each experiment**:
  trusted models (reliable PSIS-LOO and converged) with `elpd_diff >
  dse_multiplier · dse_clustered` behind the best trusted model move to
  `models/pruned/` (`dse_clustered`, `src/models/clustered_se.py`: the
  trial-level `dse` is ~2x too small). Then `_cap_live_set` keeps at most 8
  (untrusted first, then lowest ELPD). The best trusted model always survives.
  No stacking-weight floor (stacking weights are ensemble coefficients). The
  step is `pymc_orchestrator.end_experiment`; `outer_loop/reprune.py` redoes it
  for a finished experiment at another multiplier without MCMC. Recovery sweeps
  prune at 2, the October 2026 live series at 4 (user decision 2026-09-30).
- **Ledger** (`hypothesis_ledger.py`): `model_loop/attempted_hypotheses.jsonl`
  records every slot and prune, carries across experiments, and is rendered
  into every brief as "already tried — do not re-propose". A pruned mechanism
  may return with a substantive change. **`LedgerEntry` must not gain
  fields**: `from_json` requires an exact key-set match, so a new field makes
  every inherited ledger unreadable.
- **Export / "best"**: `_best_exportable_model` (`scoring.py`) — the best model
  by ELPD-LOO rank among those with reliable PSIS-LOO, **never the softmax
  posterior argmax** (rounded, so far-behind models tie at 0.0). The same rule
  gives `history.json`'s `best_model`, the critique incumbent and `SUMMARY.md`.
  `cognitive_models/` becomes the **live set**: every survivor, not only the
  winner. "Reliable" is `src/models/loo_reliability.py`'s verdict, not arviz's
  blanket k > 0.7 flag; constant-log-likelihood trials are exempt and scored
  as exactly constant (rounding noise there makes PSIS return NaN).
- **Registry** (`src/registry/io.py`): after each experiment a **uniform prior
  over the carried set**, which the next design reads. Not stacking weights:
  those made every present model's prior ~0 (zero EIG everywhere).

### Supporting modules and rules

- **Agent isolation** (`src/runtime/agent_sandbox.py`): agents run in
  bubblewrap (jobs stop without `bwrap`), their tree read-only, only their own
  directories writable, with an allowlisted environment (the backend's own
  login, no other secrets, no `SLURM_*`). Agent trees hold only `src/` and the
  run tree (`agent_tree.exclude`). `scan_gt_name.sh --before-agents` stops a
  cell whose tree names its held-out ground truth — **never write a
  ground-truth model name into anything under `src/` that agents see.**
- **Raw-only data.** `responses.csv` carries only `sequence_a, sequence_b,
  participant_id, trial_index, chose_left` (`outer_loop/columns.py`); every
  model computes its own features (`compute_features` or `prepare_observed`
  hook). Full collected rows (with Prolific IDs) go to
  `<project>/raw_collected/`, which no agent gets; `run.py` refuses a
  collecting run whose output tree is inside the repo (set
  `AUTO_PSYCH_OUTPUT_DIR`). `tests/test_agent_prompts_raw_only.py` fails if an
  agent-facing text presents a feature column as data.
- **Design screening is recorded** in `design/screened_out.json`: models that
  cannot be predicted on the pool (bookkeeping columns, their own errors,
  undefined predictions) are left out of that design on record. Person-level
  models are predicted as 40 new participants (user decision 2026-09-30;
  `docs/person_level_models.md`). Missing stimulus columns raise
  (`MissingStimulusColumns`).
- `src/models/mcmc_defaults.py` is the **single source of sampler defaults**.
- `src/models/pymc_inference.py`: fits cached on (model sha, csv sha, sampler
  settings) in-process and as `<name>.<fingerprint>.nc`; concurrent fits in a
  process pool sized to the allocated CPUs; a wave's candidates prefit in
  time-limited children before sequential admission. **An infrastructure
  failure is never a model's** (`INFRASTRUCTURE_ERRORS`,
  `FitInfrastructureFailure`): it raises, never a drop or a rejection. Each
  fit process gets its own cache and PyTensor compile directory.
- `src/model_comparison/`: ELPD-LOO softmax posterior (overconfident, a report
  field), the `az.compare` table, grouped CV (`cv.py`).
- `src/critique/ppc.py`: agent-written test statistics scored against
  posterior-predictive replicates (two-sided p + BH-FDR q). The critique runs
  **before** the round's candidates (its `critiques.md` goes into every brief);
  no usable statistic after one retry → the round runs uncritiqued and records
  `"no_critique"`. There is no fallback battery.
- `src/runtime/coding_agent.py`: `run_coding_agent` is the single launcher;
  backend = explicit arg → `CODING_AGENT` → `opencode`.
- `src/subjective_randomness/incumbent.py`: the **incumbent record** (did the
  exported best model change, is it discovered), the primary metric for
  improving the loop; `incumbent_report.py` summarises a sweep.
- Sweep summaries (`sweep_cells.py`, `reporting.aggregate_holdout_trajectories`)
  list partial and missing cells, pool by position within an experiment, and
  average the loop and every baseline over the same cells.

### Projects vs. the research library

- `src/pipelines/outer_loop/projects/<id>/`: **assets the pipeline consumes**
  (`problem_definition.md`, `task_description.md`, `seed_models/`, a gitignored
  per-run `prolific_config.yaml`). Adding a project = adding an asset dir.
- `src/subjective_randomness/`: a **harness-only research library** (agent
  trees exclude it): the PyMC model families, pair pools, recovery harness.
  `pymc_model_families/` is the frozen recovery registry, and the project's
  `seed_models/` mirrors it **byte for byte** (a test checks;
  `sync_seed_models.py` copies). Editing a seed or registry file changes its
  hash, hence every cached fit, and what agents read. Ground-truth generating
  parameters live in the recovery configs' `seed_params` (impossible models'
  explicitly in `gt_models`).
- `src/tasks/`: the `TaskSpec` per project (design pool, stimulus rows,
  prediction variable, lens battery) that keeps the pipeline generic.

### The number game

`projects/number_game/`, `src/tasks/number_game.py`; three seeds (Tenenbaum
2000 and two ports of Bigelow's numpyro models); the dataset
(`data/number-game/`, gitignored) is read by `src/number_game/`. Recovery:
`scripts/number_game/slurm/submit_recovery.sh`; `data_source: dataset` runs
draw responses from the dataset instead (`submit_dataset_run.sh`). These runs
use `fit.method: warm_start_nuts`, grouped-CV selection (`inner_loop.selection`)
and `hide_participant_id`.

### Inspecting results

`src/viewer/` (`python -m src.viewer.server`; `freeze.py` makes a static site);
`src/monitor/` is the live dashboard for an in-progress human study (first job:
catching participants who answer one side every trial).

## RSA reference games — `src/rsa/`

Rational-speech-act models of reference games (Frank & Goodman, 2012). A
**parallel stack** (user decision: keep it separate from the TaskSpec
pipeline) reusing the shared runtime, Prolific, Firebase functions and
`src/critique/ppc.py`. **Start at `docs/auto_rsa/CAMPAIGN_LOG.md`** (every
decision and run); `LOOP_ALGORITHM.md` spells out the live loop.

- Models are memo files (`src/rsa/model_file.py`) fitted with numpyro NUTS
  (`src/rsa/fit.py`); selection by grouped cross-validation; candidate agents
  have **no network** (`src/rsa/loop/no_network.py`); self-check
  `python -m src.rsa.loop.check_candidate`.
- **Data**: schema and coding decisions in
  `src/pipelines/outer_loop/projects/rsa_reference/data/README.md`, licences in
  `docs/auto_rsa/DATASETS.md`. **Never commit trial-level data from an
  unlicensed or NC-ND source** (aggregates are fine); those are fetched by
  `src.rsa.ingest` into gitignored `data/rsa/external/`.
- **The pre-registered final experiment** (`src/rsa/prereg/`,
  `docs/auto_rsa/PREREG_FINAL.md`, OSF https://osf.io/s48jv, tag
  `rsa-prereg-final`). **Never change the frozen files**
  (`data/rsa/prereg/design/`, `data/rsa/live/finals_pool_preds_final.json`,
  `data/rsa/live/bar_pool_preds.json`, `src/rsa/prereg/analysis.py`; sha256s on
  OSF) **and don't move `src/rsa/prereg/`** (the bundle hashes code by path).
  Its data are spent: nothing chosen after seeing them may be judged on them
  again (`docs/auto_rsa/CONSOLIDATION.md` §5).
- `data/rsa/` holds run records: never Prolific IDs or trial-level live data.
- Cluster scripts: `scripts/rsa/slurm/`, live launchers `scripts/rsa/live/`.

## Cluster & live runs

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

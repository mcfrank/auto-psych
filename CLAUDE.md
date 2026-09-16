# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An automated cognitive-science discovery pipeline. Coding agents conjecture PyMC
models of human judgment (currently *subjective randomness*), design maximally
informative experiments, run them on real (Prolific + Firebase) or simulated
participants, and let Bayesian model comparison decide which hypothesis survives.
`README.md` is the authoritative user-facing guide; this file captures the
architecture and workflow facts that require reading multiple files.

## Environment & commands

`uv` drives everything. The default sync installs the `dev` **and** `pymc`
groups (see `pyproject.toml` `default-groups`) — the test suite and inner loop
need the PyMC stack, so a plain `uv sync` is enough; `open-models` (torch/
transformers) is heavy and opt-in.

```bash
uv sync                      # dev + pymc (everything the tests need)
uv run pytest -q             # full suite incl. real-MCMC tests (~2 min)
uv run pytest -q -m "not slow"          # fast suite (~30 s), skips NUTS sampling
uv run pytest tests/test_eig_selection.py::test_name   # single test
./scripts/test               # wrapper: full suite (run before committing)
./scripts/test_fast          # wrapper: fast suite (run after each change)
```

- The `slow` marker means "runs PyMC NUTS". Deselect it for fast iteration.
- CI (`.github/workflows/tests.yml`) runs `uv sync --locked`, then
  `tests/test_python_sources_compile.py` (byte-compiles every `.py`), then the
  fast suite. There is **no committed linter/formatter config** (the `.ruff_cache`
  is from ad-hoc runs); match surrounding style.
- **Do not move off Python 3.11** (`.python-version`). 3.12+ resolves arviz 1.x /
  pymc 6.x, whose API rename breaks the model-comparison code. The numpy/scipy/
  pandas version caps in `pyproject.toml` exist so wheels resolve on Sherlock's
  glibc 2.17 nodes — don't drop them unless you only target glibc ≥ 2.28.

## Conventions (from the user's global instructions — they apply here)

- Filepaths via `pyprojroot.here()`; CLIs are `tyro` dataclasses (`--help` lists
  every knob). Every entry point is `python -m src.<...>` run under `uv run`.
- **Fail loudly.** This codebase deliberately raises on missing models, corrupt
  registries/artifacts, absent credentials, etc. rather than silently falling
  back. Preserve that — never add a silent default or fallback.
- Follow BDD dual-loop TDD: a failing integration test first, then inner
  red-green-refactor. Run the relevant test after each green step, the full
  suite before a commit-worthy checkpoint.

## Architecture — the mental model

Two nested loops. **The single most important fact: stages are decoupled
processes that pass state through on-disk artifacts (CSV / JSON / YAML / `.py`
files), not in-memory objects.** Every cognitive model is a self-contained `.py`
file discovered via a `models_manifest.yaml`; MCMC fits are content-addressed and
cached; the only state crossing an experiment boundary is (a) the carried model
files (the live set), (b) the ledger of attempted hypotheses beside them and
(c) the registry (a uniform prior over the carried set).

### Outer loop — `src/pipelines/outer_loop/` (`run.py` → `orchestrator.py`)

Per experiment, stages run in order. `AGENT_KEYS = ["2_design", "3_implement",
"4_collect", "5_model_loop"]` — **there is no stage 1** (the theorist and design
agents were removed; the numbering is historical). Before the stages, a
pseudo-stage establishes the model set: experiment 1 seeds from
`projects/<id>/seed_models/`; experiments ≥2 `carry_forward_cognitive_models`
from the previous experiment. Both seed/carry steps are idempotent, which is what
makes `--resume` (run into an existing `experimentN/` dir) and `--agent <stage>`
(run one stage) safe.

- `2_design` — **programmatic, no agent.** `eig.design_exhaustive` enumerates the
  H/T pair space and greedily selects the max-joint-EIG stimulus set → `design/stimuli.json`.
- `3_implement` — the one true coding-agent stage: writes a jsPsych experiment;
  skipped in `simulated_participants_nobrowser` mode. Optional Firebase/Prolific
  deploy phase follows when `--deploy-target != none`.
- `4_collect` — programmatic: writes `data/responses.csv` (simulated / LLM-as-
  participant / live / ground-truth).
- `5_model_loop` — drives the inner loop, then writes `model_registry.yaml`.

Coding stages that fail their validator are re-spawned with the error injected as
repair feedback (`--max-validation-repairs`); programmatic stages fail terminally.
Validators (`_validate_*` in `orchestrator.py`) enforce the artifact contracts
(e.g. jsPsych button-only + `chose_left` data column).

### Inner loop — `src/pipelines/inner_loop/pymc_orchestrator.py`

The only place new hypotheses enter. The model zoo lives at `model_loop/models/`;
the project's seeds are `protected_names` (never pruned — the outer loop passes
them explicitly, so a model carried from an earlier experiment *can* lose and
leave). Each round: optional CriticAL critique → spawn candidate agents in
parallel (each steered by a rotating exploration "lens") → admit sequentially.

- **Novelty gate** (`_admit_candidate`): a candidate is admitted only with a
  loadable `candidate.py` (module-level `model: pm.Model`) + `hypothesis.md` +
  `model_name.txt`, passing logp/real-fit/finite-ELPD gates, AND with posterior-
  mean `p_left` ≥ `novelty_rmse_threshold` (0.02) RMSE from every admitted model.
- **Pruning** (`_prune_losers`): non-protected, PSIS-LOO-reliable models
  statistically distinguishable from the best (`elpd_diff > dse_multiplier·dse`)
  move to `models/pruned/`. Stacking weight is deliberately not a criterion
  (it is an ensemble coefficient, not plausibility). The survivors are the
  uncertainty set — everything still within the margin of the best.
- **Ledger** (`src/pipelines/inner_loop/hypothesis_ledger.py`):
  `model_loop/attempted_hypotheses.jsonl` records every candidate slot
  (admitted / rejected, with the reason) and every prune (with the margin),
  continues the ledger the previous experiment carried, and is rendered into
  every candidate brief as `attempted_hypotheses.md` ("already tried — do not
  re-propose": the retired hypotheses, i.e. those no longer in the set).
  Without it, pruned hypotheses vanished from `existing_hypotheses.md` and were
  re-proposed (in the weakest recovery cell 11 of 13 re-proposals had already
  been pruned there).
- **Export**: the exported winner (`_best_exportable_model`) is the best model
  by **ELPD-LOO rank** (`az.compare`'s `rank`) among those whose PSIS-LOO is
  *reliable* — never the softmax posterior argmax: the posterior is rounded to
  six decimals, so every model more than ~14 nats behind reads 0.0 and a
  `max` over it picked by manifest order (62 of 230 baseline experiments
  exported a far-behind seed that way). The same rule selects the per-step
  `best_model` in `history.json` (which also records `argmax_model` and
  `excluded_unreliable`) and the critique incumbent, so what the recovery
  harness scores, what the critic critiques and what is carried agree. The
  outer loop (`_export_inner_loop_models`) then makes `cognitive_models/` the
  **live set**: the protected seeds plus every zoo survivor (not only the
  winner — a rival within 2·dse is carried and left to the next design), with
  a carried model the loop pruned removed, and the ledger copied beside the
  manifest. Before this, only the winner crossed the boundary: 19 unresolved
  rivals were dropped at 40 boundaries in the iteration-2 recovery sweep.
  "Reliable" is `src/models/loo_reliability.py`'s verdict (a tolerated
  proportion of high-Pareto-k trials, with constant-log-likelihood trials
  exempt as exact), **not** arviz's blanket any-k>0.7 flag — that flag fired on
  clipped `p_left` trials and was silently discarding genuine winners.

### How weights flow between experiments (the registry)

`src/registry/io.py` schema: `{theories: {name: prob}, reserved_for_new}`. After
experiment N, `update_registry_from_interpretation` writes a **uniform prior
over the carried set** (the `cognitive_models/` manifest the export just
wrote) to `model_registry.yaml`; experiment N+1's design reads it as the model
prior for EIG selection. It used to copy `az.compare`'s stacking weights, which
are ensemble coefficients rather than plausibility and were written over the
whole zoo: in 15 of 40 next-experiment designs of the iteration-2 recovery
sweep every model actually present had weight ~0 (or one had 1.0), so all 32
EIG-selected stimuli had zero EIG. The stacking weights remain a report field
in `model_posterior.json`. Model *files* flow separately via carry-forward.

### Supporting modules

- **Screening a model out of a design is narrow and recorded.**
  `eig._screen_usable_models` may omit a model only when the columns it cannot
  bind are response-row bookkeeping (`NON_STIMULUS_COLUMNS`: `participant_id`,
  `trial_index`) — a participant-level random effect, say. Missing *feature*
  columns raise instead: that means the design rows were built without the
  featurizer the models read, and dropping the model would renormalize EIG over
  whichever ones happen to bind. `make_stim_data` signals this with
  `MissingStimulusColumns`, which carries `.missing` as data so callers classify
  structurally rather than by re-parsing a message. Every drop is written to
  `design/screened_out.json` (empty list = the screen ran and dropped nothing),
  so a silent shrink of the hypothesis set is visible in the run tree.
- **Two seed sets, one per feature regime.** `pymc_model_families/` (and the
  live pool `seed_models/`) bind the columns the project featurizer supplies;
  `pymc_model_families_raw/` (pool `seed_models_raw/`) compute those columns
  themselves and are used only by a `raw_features: true` config, which writes an
  agents' CSV of H/T sequences alone and designs without a featurizer. The sets
  must never be merged: a model computing a column the CSV already carries is
  unfittable, and the inner loop `[drop]`s it rather than failing. Why the arm
  exists (the featurizer reproduces each ground truth's own decision variable at
  R² 0.90-1.00) and how to read its results: `docs/raw_features_arm.md`.
- `src/models/mcmc_defaults.py` — the **single source of MCMC sampler defaults**
  (`PRODUCTION_*`, `DESIGN_TWIN_*`). Every entry point imports from here; change
  defaults only here.
- `src/models/pymc_inference.py` — bridge from agent-written `.py` files to
  inference. `load_pymc_model` requires a module-level `model: pm.Model`. Two
  optional model hooks: `compute_features(seq_a, seq_b)` or `prepare_observed(rows)`
  (mutually exclusive). Fits are cached on `(model sha, csv sha, sampler sig)`
  in-process and on disk (`<name>.<fingerprint>.nc`).
- `src/model_comparison/{posterior,likelihood}.py` — ELPD-LOO softmax posterior
  (`model_posterior`, documented as overconfident) + `az.compare` PSIS-LOO table.
- `src/critique/ppc.py` — CriticAL posterior-predictive check: agent-written
  `test_statistic(df) -> float` scored against posterior-predictive replicates,
  two-sided empirical p + BH-FDR q; results steer the next candidate round.
- `src/runtime/coding_agent.py` — backend-agnostic agent launcher.
  `run_coding_agent(...)` is the single call site; `select_backend` resolves
  explicit arg → `CODING_AGENT` env → `opencode` default. `claude` uses
  `--dangerously-skip-permissions --add-dir`; `opencode` uses `opencode run`
  (no `--add-dir`). Token usage is always recorded.

### Projects vs. the research library — two different things

- `src/pipelines/outer_loop/projects/<id>/` = **assets the generic pipeline
  consumes** (`problem_definition.md`, `preprocess.py` featurizer,
  `ground_truth_models.py`, `seed_models/`, `prolific_config.yaml`). Adding a
  project = adding an asset directory. This lives under `src/`, not the
  run-output `projects/` tree.
- `src/subjective_randomness/` = a **standalone research library** for the
  subjective-randomness domain (model families, `stimulus_design.py`,
  `sequence_stats.py`, recovery harnesses). Coupling to the pipeline is
  deliberately thin (two cross-imports). `pymc_model_families/` is the frozen
  recovery registry; the project's `seed_models/` manifest mirrors it (a test
  asserts they agree).

### Inspecting results

- `src/viewer/` — Flask + static SPA explorer over finished runs on disk
  (`python -m src.viewer.server`); `freeze.py` snapshots curated runs to a static
  site. The tree is re-walked per request (no build step).
- `src/monitor/` — live dashboard for an in-progress human study (Firestore +
  Prolific), discovered from `deployment_manifest.json` files. Its first job is
  catching degenerate data (participants answering one side every trial).

### Legacy / not part of the live loops

`src/experiments/` and `src/validation/` are from the old pipeline and are not
used by the active loops (per `README.md`).

## Cluster & live runs

- **Live runs recruit real participants and spend real money.** They are double-
  gated: the config needs `confirm_live_recruitment: true` **and** `run.py`
  enforces `--confirm-live-recruitment`; the launchers print a cost summary and
  require typing `yes`. See `scripts/outer_loop_live/README.md`. `scancel` kills
  the pipeline job but **not** an already-published Prolific study — stop that in
  the Prolific dashboard.
- Secrets live in repo-root `.secrets` (see `.secrets.example`): `PROLIFIC_API_TOKEN`,
  `FIREBASE_TOKEN` (`firebase login:ci`), `AUTO_PSYCH_RESULTS_TOKEN` (guards the
  `/submit` & `/results` Cloud Functions — deploy/collect fail loudly without it),
  `GOOGLE_API_KEY` (simulated/Gemini paths only).
- On Sherlock: never run heavy work on the login node (submit via Slurm), keep
  job I/O on `$SCRATCH`, and note Playwright browser simulation does **not** run
  on the compute nodes (glibc 2.17) — use `--mode simulated_participants_nobrowser`.
  Recovery-harness runbooks: `scripts/subjective_randomness/README.md` and its
  `slurm/README.md`.

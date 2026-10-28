# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An automated cognitive-science discovery pipeline: coding agents conjecture
models of human judgment, EIG picks the experiments, real (Prolific + Firebase)
or simulated participants answer, and Bayesian model comparison decides which
hypothesis survives. Domains: *subjective randomness* and the *number game*
(shared PyMC pipeline) and *RSA reference games* (`src/rsa/`, memo + numpyro, a
separate stack). `README.md` is the user guide; the details behind this file
are in `docs/architecture/` (table at the end).

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

- The full suite takes over an hour (1:13 on 8 CPUs) and needs more than 5 GB of memory (the
  JAX/numpyro RSA tests); on a small interactive allocation it dies silently.
- CI (`.github/workflows/tests.yml`): `uv sync --locked`,
  `tests/test_python_sources_compile.py`, then the fast suite. No committed
  linter/formatter config; match surrounding style.
- **Python 3.12** (memo needs ≥ 3.12). **Keep the `arviz<1` / `pymc<6` caps**
  (newer releases rename the API the model comparison uses) and the
  jax/jaxlib 0.7.0, numpyro 0.19.0 and numpy/scipy/pandas pins: they keep wheels
  resolving on Sherlock's glibc 2.17 nodes.

## Conventions

- Filepaths via `pyprojroot.here()`; CLIs are `tyro` dataclasses (`--help` lists
  every knob), run under `uv run`.
- **Fail loudly.** The codebase deliberately raises on missing models, corrupt
  artifacts, absent credentials, etc. Never add a silent default or fallback.
- BDD dual-loop TDD: a failing integration test first, then red-green-refactor.

## Architecture in one screen

Two nested loops. **Stages are decoupled processes that pass state through
on-disk artifacts (CSV / JSON / YAML / `.py` files), not in-memory objects.**
Every cognitive model is a self-contained `.py` file listed in a
`models_manifest.yaml`; MCMC fits are content-addressed and cached. The only
state crossing an experiment boundary: the carried model files (the live set),
the ledger of attempted hypotheses beside them, and the registry (a uniform
prior over the carried set).

- **Outer loop** (`src/pipelines/outer_loop/`, `run.py` → `orchestrator.py`):
  `2_design` (programmatic max-joint-EIG selection over the task's whole pool;
  after experiment 1 it loads the previous model loop's own fits and samples
  nothing) → `3_implement` (the one coding-agent stage: a jsPsych page that
  shuffles the trial order per participant) → `4_collect` (`data/responses.csv`)
  → `5_model_loop` (the inner loop, then export and registry). There is no
  stage 1.
- **Inner loop** (`src/pipelines/inner_loop/`): each round runs a critique, then
  candidate agents in parallel, then sequential admission through the code,
  logp, data-contract, fit, convergence, time-limit and novelty gates. A
  rejected candidate gets one repair. Pruning runs once, at the end of each
  experiment.

## Rules that span files

- **Never write a ground-truth model name into anything under `src/` that
  agents see** (`scan_gt_name.sh --before-agents` stops the cell).
- **Raw-only data:** `responses.csv` carries only `sequence_a, sequence_b,
  participant_id, trial_index, chose_left`; every model computes its own
  features. Prolific IDs go only to `<project>/raw_collected/`, which no agent
  gets.
- **`LedgerEntry` must not gain fields** (`from_json` requires an exact key-set
  match, so a new field makes every inherited ledger unreadable).
- A project's `seed_models/` mirrors `src/subjective_randomness/pymc_model_families/`
  **byte for byte** (`sync_seed_models.py`); editing either changes every cached
  fit and what agents read.
- `src/models/mcmc_defaults.py` is the **single source of sampler defaults**.
- **An infrastructure failure is never a model's** (`INFRASTRUCTURE_ERRORS`):
  it raises, never a drop or a rejection.
- **"Best"** is the top ELPD-LOO rank among models with reliable PSIS-LOO
  (`_best_exportable_model`), never the softmax posterior argmax.
- Starting models are not protected: they are pruned like any model.
- **RSA pre-registration:** never change the frozen files or move
  `src/rsa/prereg/` (list in `docs/architecture/rsa.md`).
- Never commit Prolific IDs, trial-level live RSA data, or trial-level data from
  an unlicensed or NC-ND source.
- **Live runs recruit real people and spend real money:** double-gated
  (config **and** `--confirm-live-recruitment`); `scancel` does not stop a
  published Prolific study.
- Claude billing is stated per run (`CLAUDE_AUTH` / `--claude-auth`), never
  guessed.
- Never run heavy work on a login node; keep job I/O on scratch.

## Where the details are

| topic | file |
|---|---|
| outer-loop stages, design, restarts, validators | `docs/architecture/outer_loop.md` |
| admission gates, retry/repair, slot roles, pruning, ledger, export, registry | `docs/architecture/inner_loop.md` |
| sandbox, raw-only data, inference, comparison, critique, projects, number game, the sr-oct26 and number-game ablations, viewer | `docs/architecture/supporting_modules.md` |
| RSA stack and pre-registration | `docs/architecture/rsa.md` (start at `docs/auto_rsa/CAMPAIGN_LOG.md`) |
| clusters, sweeps, live-study rules, secrets, billing | `docs/architecture/cluster_and_live_runs.md` |

# Auto-psych — Code Summary

This repository implements an **automated psychology research loop**: a system in
which LLM "coding agents" (Claude Code or opencode) and statistical machinery
together propose cognitive theories, design stimuli, build and deploy a web
experiment, collect data (from synthetic models, an LLM-as-participant, or real
humans on Prolific), and then iteratively fit, criticize, and improve
probabilistic cognitive models of that data.

The system is organized as **two nested loops**:

- **Outer loop** (`src/pipelines/outer_loop/`) — the *experiment* loop. One pass
  proposes models, designs stimuli, implements + deploys an experiment, collects
  data, and hands the data to the inner loop. Multiple experiments can be chained
  so each builds on the previous one's posterior.
- **Inner loop** (`src/pipelines/inner_loop/`) — the *model-improvement* loop. It
  fits a set of PyMC cognitive models to the collected responses, ranks them by
  ELPD-LOO, runs a posterior-predictive **critique** (CriticAL) of the best
  model, asks a coding agent to propose new candidate models that fix the
  critique's discrepancies, and exports the winning model back to the outer loop.

The only fully built-out scientific project is **`subjective_randomness`** (how
people judge which of two H/T coin sequences "looks more random").

---

## 1. Outer loop — the experiment pipeline

Entry point: `python -m src.pipelines.outer_loop.run` (`src/pipelines/outer_loop/run.py`).
The orchestration helpers live in `src/pipelines/outer_loop/orchestrator.py`.

A run iterates over one or more experiments. For each experiment it runs an
ordered list of **stages**, each driven either by a spawned coding agent (reads a
prompt from `prompts/<stage>.md` + a generated `CONTEXT.md`) or by programmatic
Python:

| Stage | Driver | What it does |
|---|---|---|
| `1_theory` | coding agent (or seeded) | Writes PyMC cognitive models (`cognitive_models/<name>.py`) + `models_manifest.yaml`, each model carrying a one-sentence hypothesis ("rationale"). Experiment 1 can instead **seed** models from `projects/<project>/seed_models/`. |
| `2_design` | coding agent **or** `exhaustive` | Selects stimuli into `design/stimuli.json`. `--design-mode exhaustive` replaces the agent: it enumerates the full H/T pair space and greedily picks a diverse, jointly-informative set by expected information gain (EIG). For experiments ≥2 the EIG is computed under the *previous* experiment's posterior. |
| `3_implement` | coding agent | Writes a jsPsych browser experiment (`experiment/index.html` + `config.json`). Skipped in `simulated_participants_nobrowser` mode. |
| *(deployment)* | programmatic | After implement, optionally deploy to Firebase Hosting + create a Prolific study (`--deploy-target`, `--prolific-mode`). |
| `4_collect` | programmatic | Collects responses → `data/responses.csv` (see collection modes below). |
| `5_model_loop` | programmatic | Runs the inner loop over pooled responses; exports the best model. |

Stage output is optionally checked by `validate_cc_output` (per-stage validators
in `orchestrator.py`, e.g. `_validate_theory` builds each model graph, `_validate_implement`
enforces a jsPsych button-response data contract and that every design stimulus
appears verbatim in the HTML).

After `5_model_loop`, `update_registry_from_interpretation` records the inner
loop's actual posterior over models into the experiment's `model_registry.yaml`
(used to weight the posterior-aware EIG design of a later experiment).

### Collection modes (`--mode`)

`run_collect_programmatic` (`orchestrator.py`) chooses a source:

- **`simulated_participants`** (default) — synthetic data sampled from the
  theorist's PyMC models' prior-predictive `p_left` (`_generate_from_pymc_models`),
  or from a named `--ground-truth-model` callable (`_generate_from_models`).
- **`simulated_participants_nobrowser`** — **LLM-as-participant**: each synthetic
  participant answers every stimulus directly through a language model
  (`participants.py`: `ClosedParticipantModel` via a Gemini/LangChain client, or
  `OpenParticipantModel` via a local HuggingFace model). The browser/jsPsych
  stage is skipped.
- **`live`** — real participants. Reads results from the deployed experiment's
  results API / Firestore (`_collect_live`, `_collect_from_firebase`).

A degenerate-data quality gate (`check_response_variation`) is meant to reject
data where everyone chose the same side.

### Deployment (`src/pipelines/outer_loop/deployment/`)

`run_deployment` orchestrates: `firebase.py` (writes a per-run Firebase Hosting
config, injects a consent gate + a `/submit` bridge to a Cloud Function, runs
`firebase deploy`), `firestore.py` (writes study/deployment/session metadata
docs), `prolific.py` + `runtime/prolific.py` (creates/publishes a Prolific
study, computes reward), `manifest.py` (writes a `deployment_manifest.json` that
the monitor later discovers), `local.py` (the dry-run/local path), and `smoke.py`
(a tiny synthetic experiment for smoke-testing deploys). `--run-label` isolates
parallel runs into distinct `/e{N}-{label}/` hosting paths.

---

## 2. Inner loop — PyMC model-improvement loop

Entry point: `python -m src.pipelines.inner_loop.run`; core logic in
`src/pipelines/inner_loop/pymc_orchestrator.py::run_pymc_inner_loop`.

Flow:

1. **Seed** the model "zoo" from `seed_models_dir` (`_seed_model_set`), then drop
   any model whose log-density is non-finite on the data (`_drop_unfittable_models`).
2. **Score** the set: fit every model by MCMC (`src/models/pymc_inference.py::fit_model`,
   cached on disk as `.nc` + in-process), compute a Bayesian **posterior over
   models** as a softmax of ELPD-LOO plus an optional complexity (Occam) prior
   (`src/model_comparison/posterior.py::model_posterior`).
3. For each candidate **round** (`max_iterations`):
   - **Critique** the incumbent (best) model (`_run_critique_round`): seed its
     fit into a cache, spawn a critique agent that writes test statistics under
     `test_stats/`, then run the **posterior-predictive check** harness
     (`src/critique/ppc.py`) in-process. Each statistic is evaluated on the
     observed data vs. `CRITIQUE_PPC_REPLICATES` posterior-predictive replicate
     datasets, producing a two-sided empirical p-value plus a Benjamini-Hochberg
     FDR-adjusted q across the round's statistics; statistics significant at raw
     `alpha` are flagged in `critiques.md` (FDR-surviving ones called out, since
     this is exploratory screening of several statistics at once).
   - **Generate** `candidate_count` new models: spawn coding agents
     (`_spawn_candidate_agent`), each told to propose **one** distinct/refined
     single-mechanism hypothesis (never a blend), prioritizing the critique's
     discrepancies. Admit each via `_admit_candidate` (must load as a PyMC model,
     have finite logp, and ship a `hypothesis.md`).
   - **Re-score** the enlarged set; append to `history.json`.
4. **Export** (`_export`): `model_posterior.json`, `best_model.py`, a `report.md`
   with the posterior table + an `arviz.compare` PSIS-LOO distinguishability
   table, and the per-round history.

`src/models/pymc_inference.py` is the inference workhorse: dynamic loading of
agent-written model files (`load_pymc_model` via `importlib`), `extract_observed`,
`fit_model`/`fit_models_cached` (MCMC + caching), `model_logp_is_finite`,
posterior-predictive sampling (`predict_p_left`, `sample_synthetic_responses`),
and posterior thinning. `src/model_comparison/` holds the ELPD-LOO posterior,
the `az.compare` table, and a pooled-CSV likelihood helper.

---

## 3. The `subjective_randomness` project

Two parallel locations hold this project's code:

- **`src/pipelines/outer_loop/projects/subjective_randomness/`** — pipeline
  *assets*: `problem_definition.md`, `ground_truth_models.py`, `preprocess.py`
  (the `featurize_stimulus` that turns H/T strings into numeric feature columns),
  `seed_models/` (the PyMC seed model set), reference PDFs.
- **`src/subjective_randomness/`** — a large standalone research library for
  parameter/model recovery experiments and stimulus selection, largely
  independent of the agent pipeline. Highlights:
  - `model_families/` + `pymc_model_families/` — four cognitive model families
    (prototype similarity, Bayesian diagnosticity, statistical inference,
    encoding compressibility), as pure-Python predictors and as PyMC models.
  - `features.py` — sequence features (alternation rate, run statistics,
    periodicity, compressibility, etc.).
  - `stimulus_design.py` — exhaustive H/T pair enumeration + greedy EIG selection
    (`build_exhaustive_design`, `select_informative_stimuli`) used by the outer
    loop's exhaustive design mode.
  - `model_recovery.py`, `recover.py`, `pymc_recover.py` — generate data from a
    known model/params and check the pipeline recovers them.
  - `holdout_recovery.py`, `adaptive_recovery.py` — held-out cross-validation and
    an **EIG-vs-random** adaptive-design comparison (sequential Bayesian model
    evidence over a parameter grid).
  - `discriminating_probe.py`, `model_similarity_judge.py`, `impossible_models/` —
    adversarial/discriminating stimulus selection and an LLM judge of model
    similarity; "impossible" models used as recovery stress tests.

---

## 4. Viewer, Monitor, and Freeze

- **Viewer** (`src/viewer/`, `python -m src.viewer.server`) — a Flask + static
  SPA run explorer. `scan.py` walks `data/`, finds runs (dirs containing
  experiments or a bare model loop), and builds structured payloads
  (`models.py`) covering seed models, design, deployed experiment, collected
  data, the inner-loop trajectory + model comparison, and the critiques.
  `transcripts.py` strips ANSI from agent logs. The data tree is re-scanned per
  request (no build step).
- **Freeze** (`src/viewer/freeze.py`, `python -m src.viewer.freeze`) — calls the
  same scanners to produce a self-contained static JSON snapshot of curated runs
  for hosting on Firebase (read-only public viewer).
- **Monitor** (`src/monitor/`, `python -m src.monitor.server`) — a separate live
  dashboard for in-progress human studies. `discovery.py` finds studies from
  `deployment_manifest.json` files; `sources.py` reads live submissions from
  Firestore and recruitment status from Prolific; `aggregate.py` computes
  per-participant and choice-balance stats and flags degenerate data (everyone
  choosing one side); `report.py` assembles the API payload. Auto-refreshes
  every 15 s.

---

## 5. Supporting modules

- `src/runtime/` — `coding_agent.py` (backend-agnostic subprocess launcher for
  Claude Code / opencode, streaming output to a `.jsonl` log), `config.py`
  (`REPO_ROOT` etc.), `prolific.py` (Prolific REST client), `console.py`,
  `observability.py`.
- `src/registry/io.py` — per-run `model_registry.yaml` (theory → probability,
  plus `reserved_for_new`).
- `src/validation/` — stage validators (`stages/*.py`) and a `validators.py`
  wrapper used to record validation failures.
- `src/experiments/` — experiment `state`/`state_loader`, `problem_definition`
  parsing, `references.py` (assembles reference text from PDFs).
- `src/models/theorist/` — loads agent-written prediction callables and gathers
  their predictions.

---

## 6. Data layout, deployment config, and tests

```
data/outer_loop/<project>/experiment<N>/
    cognitive_models/   design/   experiment/   data/   model_loop/
    model_registry.yaml CONTEXT.md logs/
```

`model_loop/` holds `models/`, `model_posterior.json`, `history.json`,
`best_model.py`, `report.md`, and per-iteration `critique/` artifacts. Firebase
config (`firebase.json`, `firestore.rules`, `functions/`) and a separate
hosting-only viewer config (`firebase.viewer.json`) live at the repo root.
Secrets are read from environment or a git-ignored `.secrets` file.

Tests live under `tests/` and `src/pipelines/inner_loop/tests/`, run with
`uv run --group dev pytest tests src/pipelines/inner_loop/tests -q`.

> See `PROBLEMS.md` for the code review and the list of fixes applied during it.

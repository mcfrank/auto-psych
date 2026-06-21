# Auto-psych

Active development is organized around two explicit loops:

- `src/pipelines/outer_loop`: experiment loop. Coding agents propose models, design stimuli, implement experiments, collect data, then hand observed data to the inner model loop.
- `src/pipelines/inner_loop`: cognitive-model improvement loop. It fits and compares candidate **PyMC** models by ELPD-LOO, exports the best model, and maintains the model-zoo/posterior artifacts. Before each candidate round it runs a CriticAL posterior-predictive critique (`src/critique`) of the incumbent best model and feeds the resulting `critiques.md` to the candidate agents.

The old LangGraph/LangChain pipeline, Cloud Run entrypoint, and SLURM submitter have been removed.

## Setup

```bash
uv sync --group dev
```

Optional PyMC support:

```bash
uv sync --group pymc
```

Optional open-weight participant models (for `--participant-backend open`; pulls
in `torch` + `transformers`, so install only when needed):

```bash
uv sync --group open-models
```

`open-models` resolves CPU / Apple-MPS `torch` by default. For an NVIDIA GPU,
install the `torch` wheel matching your CUDA version from
<https://pytorch.org/get-started/locally/> first, then run the sync above.

## Run The Active Outer Loop

```bash
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1
```

Useful options:

```bash
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiments 3
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --agent 5_model_loop
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --inner-loop-iterations 2 --inner-loop-candidates 3
```

### Collection modes

`--mode` selects how stage `4_collect` gathers responses:

- `simulated_participants` (default): synthetic data from the theorist's PyMC
  models (or a `--ground-truth-model`). No browser.
- `simulated_participants_nobrowser`: **LLM-as-participant** — each synthetic
  participant answers every stimulus directly via a language model. The jsPsych
  `3_implement` stage is skipped in a full run.
- `live`: real participants via Prolific + Firebase. Requires a deployed
  experiment to collect from (run with `--deploy-target firebase --prolific-mode
  live`); it reads submissions from the results API / Firestore and refuses to
  fall back to synthetic data if no deployment is configured.

For `simulated_participants_nobrowser`, choose the participant-model backend:

```bash
# Closed / hosted API (default: the project's Gemini client)
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness \
    --experiment 1 --mode simulated_participants_nobrowser --participant-backend closed

# Open-weight, local Hugging Face model (needs the open-models group)
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness \
    --experiment 1 --mode simulated_participants_nobrowser \
    --participant-backend open --hf-model Qwen/Qwen2.5-0.5B-Instruct
```

The closed backend needs `GOOGLE_API_KEY` (env or `.secrets`); `--closed-model`
overrides the default model id. The open backend loads the named model locally;
`--hf-model` defaults to `Qwen/Qwen2.5-7B-Instruct` (large — pass a smaller id,
e.g. `Qwen/Qwen2.5-0.5B-Instruct`, for quick runs).

**This participant model is separate from the coding-agent backend.** The
theory / design / implement stages (and inner-loop candidate generation) are
driven by `--coding-agent` (opencode by default); `--participant-backend` /
`--hf-model` only choose the model that *answers trials* during `4_collect`.

Smoke-test just the open participant path (no PyMC, no API key, tiny model):

```bash
uv sync --group open-models
uv run python scripts/smoke_open_participant.py
```

### Coding-agent backend

Both loops spawn a coding-agent CLI to write models/experiments. The default is
opencode (default model `google/gemini-3.1-pro-preview`); pass `--coding-agent
claude` (or set `CODING_AGENT=claude`) to use Claude Code (`claude-sonnet-4-6`)
instead. The backend is resolved once and exported so the inner loop inherits it.

```bash
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --coding-agent claude
```

opencode runs headless via `opencode run`; grant it edit/bash permission in
`opencode.json` (the equivalent of Claude's `--dangerously-skip-permissions`),
and confirm its default model id (`google/gemini-3.1-pro-preview`) is valid for
your install (set a different model in `opencode.json` if not — the runners have
no model-override flag).

Project *assets* (problem definition, ground-truth models, featurizer) live under
`src/pipelines/outer_loop/projects/<project>/`. Generated experiment *outputs* are
written under:

```text
data/outer_loop/<project>/experiment<N>/
```

The inner loop writes:

```text
model_loop/
model_loop/iter_<i>/critique/test_stats/*.py   # proposed test statistics
model_loop/iter_<i>/critique/ppc_results.json  # empirical + FDR-adjusted p-values
model_loop/iter_<i>/critique/critiques.md      # significant discrepancies of the incumbent
cognitive_models/inner_loop_model.py
cognitive_models/models_manifest.yaml
```

## Browse Run Results

A live web explorer presents one page per **run**. A run is any directory under
`data/` that holds experiments (or a single bare model loop) — for example
`data/outer_loop/subjective_randomness` or
`data/subjective_randomness/impossible_holdout_recovery/impossible_holdout_runs/fewer_heads_more_random`.
The sidebar is a directory tree of every run found; clicking one opens its page.

Each run page stacks its experiments as collapsible panels (the first open).
Every experiment shows: the seed cognitive models, the stimulus design, the
deployed experiment, the collected data, the inner model loop (interactive
posterior trajectory + sortable model comparison + the **models proposed at each
iteration** with hypothesis/code/transcript), and the **critiques** — every
proposed posterior-predictive test statistic with its description, code, and PPC
result (observed vs. null, FDR-adjusted p, significant discrepancy). Run-level
analysis figures (`<run>/analysis/*.png`) appear at the top.

```bash
uv run python -m src.viewer.server               # serves http://127.0.0.1:8000
uv run python -m src.viewer.server --data-root data --port 9000
uv run python -m src.viewer.server --host 0.0.0.0   # expose on the network
```

The data tree is walked on each request, so the explorer always reflects the
latest runs — no build step. Partial runs (e.g. smoke runs that skipped
modeling) load fine; corrupt JSON/YAML artifacts fail loudly with the offending
filename.

### Publish to the web (static snapshot)

To share results with collaborators, freeze a **curated** set of runs into a
self-contained static site and host it on Firebase. `src/viewer/freeze.py` calls
the same scanners the live server uses, so the frozen JSON is identical to the
API — the same frontend renders it with no server.

```bash
# 1. Freeze the runs you want to publish into viewer_dist/ (git-ignored).
uv run python -m src.viewer.freeze \
    --data-root data \
    --out-dir viewer_dist \
    --run-paths outer_loop/subjective_randomness

# 2. One-time: create the dedicated hosting site (needs `firebase login`).
firebase hosting:sites:create auto-psych-viewer --project auto-psych-2c5da

# 3. Deploy. The isolated firebase.viewer.json hosting-only config can never
#    touch the experiment site or its Cloud Functions.
firebase deploy --config firebase.viewer.json \
    --only hosting:auto-psych-viewer \
    --project auto-psych-2c5da --non-interactive
# → https://auto-psych-viewer.web.app
```

The snapshot is a public, read-only point-in-time copy (re-run the freeze + deploy
to refresh it). Only the listed `--run-paths` are published; each must be a run
the viewer discovers, or the freeze fails loudly. Note the frozen JSON inlines
participant response previews and full agent transcripts — the freeze prints what
it is publishing so you can vet it before deploying publicly.

## Monitor A Live Study

While a human study is collecting data on Prolific + Firebase, a separate
dashboard shows what is happening **in real time** — distinct from the run
explorer above, which reads finished runs from disk. The monitor reads live
participant submissions from Firestore and recruitment status from Prolific.

```bash
uv run python -m src.monitor.server                 # serves http://127.0.0.1:8001
uv run python -m src.monitor.server --port 9001
uv run python -m src.monitor.server --firebase-project auto-psych-2c5da
uv run python -m src.monitor.server --host 0.0.0.0  # expose on the network
```

It discovers studies to watch from the `deployment_manifest.json` files each
deploy writes into `data/` (only real `firebase` deployments — dry-runs are
skipped), so launching a study makes it appear with no restart. For each study
the dashboard shows completion vs. target, the Prolific recruitment counts
(active / awaiting review / approved / returned / timed out), and a
**per-participant breakdown**.

The first job of the monitor is to catch degenerate data early. It flags any
participant who chose the same side on every trial and warns loudly when the
overall left/right split collapses to one side across the study — exactly the
silent failure mode (everyone answering identically) that can quietly ruin a
pilot. The page auto-refreshes every 15 s.

Reading live data needs credentials: Application Default Credentials for
Firestore (`gcloud auth application-default login`) and `PROLIFIC_API_TOKEN`
(env or `.secrets`) for the recruitment counts. If Prolific is unreachable the
participant data still renders and the Prolific error is shown inline.

## Project Layout

```text
src/
  pipelines/
    outer_loop/
      run.py                 # `python -m src.pipelines.outer_loop.run` (entry point)
      orchestrator.py        # stage spawning, programmatic collect/deploy/inner-loop, validators
      collect.py             # collection modes (simulated / LLM-as-participant / live)
      participants.py        # closed (Gemini) + open (HuggingFace) participant models
      llm.py eig.py
      prompts/               # 1_theory / 2_design / 3_implement / 4_collect_* prompts
      projects/<project>/    # problem_definition.md, ground_truth_models.py, preprocess.py, seed_models/
      deployment/            # firebase.py firestore.py prolific.py manifest.py local.py smoke.py
    inner_loop/
      run.py                 # `python -m src.pipelines.inner_loop.run` (entry point)
      pymc_orchestrator.py   # seed/score/critique/candidate rounds, export best model
      prompts/               # pymc_theory.md + critique.md
  critique/
    ppc.py                   # CriticAL posterior-predictive check (empirical p + BH-FDR)
  model_comparison/
    posterior.py             # ELPD-LOO softmax posterior over models + az.compare table
    likelihood.py            # ELPD-LOO of one model
  models/
    pymc_inference.py        # load/fit (MCMC) agent-written PyMC models, PPC sampling, caching
    theorist/                # loader.py + predictions.py (pure-Python prediction callables)
    project/                 # ground_truth.py
  subjective_randomness/     # standalone research library: model families, recovery, stimulus design
  runtime/
    coding_agent.py          # backend-agnostic Claude Code / opencode subprocess launcher
    config.py console.py observability.py prolific.py
  experiments/
    state.py state_loader.py problem_definition.py references.py
  registry/
    io.py                    # per-run model_registry.yaml (theory -> probability)
  validation/
    validators.py stages/    # per-stage output validators
  viewer/                    # browser-based run explorer (Flask + static SPA)
    server.py                # `python -m src.viewer.server`
    freeze.py                # `python -m src.viewer.freeze` -> static snapshot for web hosting
    scan.py models.py transcripts.py static/
  monitor/                   # live dashboard for in-progress human studies (Flask + static SPA)
    server.py                # `python -m src.monitor.server`
    discovery.py sources.py aggregate.py report.py models.py static/
```

## Tests

```bash
uv run --group dev pytest tests src/pipelines/inner_loop/tests -q
```

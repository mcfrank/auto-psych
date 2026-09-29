# Auto-psych

**New here or coming back after a while?** Start with [`docs/onboarding/BRIEF.md`](docs/onboarding/BRIEF.md) (the paper's system, how the current loop differs, where it stands, and what a live experiment needs); the rest of the guide, including the live-experiment runbook, is in [`docs/onboarding/`](docs/onboarding/README.md).

An automated cognitive-science discovery pipeline. Coding agents iteratively
conjecture computational cognitive models (PyMC) of human judgment — currently
of *subjective randomness* ("which sequence looks more random?") — design
maximally informative experiments, run them on real participants (Prolific +
Firebase) or simulated ones, and let Bayesian model comparison decide which
hypothesis survives.

Active development is organized around two explicit loops:

- `src/pipelines/outer_loop`: **experiment loop**. Each experiment starts from
  a model set — seeded from the project's `seed_models/` in experiment 1
  (four literature models of subjective randomness) and carried forward
  afterwards (the previous experiment's surviving models); there is no
  theorist agent. The
  design stage is programmatic (no design agent): it enumerates the full H/T
  pair space and greedily selects (lazy batched greedy, float32, on every
  allocated CPU) the jointly most informative stimulus set by
  expected information gain (EIG) under the current model weights (experiments
  ≥2 design from the previous experiment's posterior); an implement agent
  builds the jsPsych experiment; collection gathers responses; then the
  observed data go to the inner model loop.
- `src/pipelines/inner_loop`: **model-discovery loop** — the only place new
  hypotheses enter. Each round it critiques the incumbent best model with a
  CriticAL posterior-predictive check (`src/critique`) — the critique agent
  proposes the test statistics; if it writes none it is retried once, and
  after that the round runs with no critique and `history.json` says so (the
  pipeline never substitutes statistics of its own) — then spawns candidate
  agents in parallel, one per slot (self-named via `model_name.txt`):
  exploratory slots, each steered by a distinct exploration lens, write one
  new single-mechanism PyMC model apiece; refinement slots improve a model
  already proposed — two the incumbent, named in the brief, and one a live or
  pruned model of the agent's choosing from a menu with full hypotheses.
  A slot whose agent writes nothing is retried once; a candidate rejected at
  admission is repaired once, with the rejection reason in the agent's
  prompt; every attempt is in the ledger, and the agent can run a documented
  self-check (a smoke fit) before it finishes.
  Candidates are admitted only if they fit by MCMC, achieve finite ELPD-LOO,
  and are **genuinely novel** (a candidate predicting within 0.002 RMSE of an
  existing model's `p_left` on a broad, loop-generated stimulus pool — not the
  training stimuli — is rejected as a duplicate), whose MCMC converged, and
  whose admission fit finishes each sampling run within 30 minutes.
  At the end of each experiment, models that are statistically
  distinguishable losers are pruned — the starting models included, since
  2026-09-28 (earlier runs never pruned them; `starting_models.json` and
  `holdout.json` record which rule a run used) — and the live set is capped
  at 8 models. The live set — the winner and every
  model still within the margin of the best — is carried into
  `cognitive_models/`, and the next experiment's design uses a uniform model
  prior over it.

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

## Entry Points

Everything below is run with `uv run` from the repo root. Each CLI is a `tyro`
dataclass — `--help` lists every knob with its documented default.

**The pipeline**

| Command | What it does |
|---|---|
| `python -m src.pipelines.outer_loop.run` | The main pipeline: per experiment, seed/carry-forward the model set → `2_design` (programmatic exhaustive EIG selection) → `3_implement` (+ Firebase deploy when `--deploy-target firebase`) → `4_collect` → `5_model_loop`. `--agent <stage>` reruns one stage; `--resume` continues an existing tree. |
| `python -m src.pipelines.inner_loop.run` | The inner model loop standalone, on an already-featurized responses CSV + a seed-model dir. Exposes all discovery knobs (`--hints-file`, `--novelty-rmse-threshold`, `--prune-*`, `--candidate-parallelism`). |
| `python -m src.pipelines.outer_loop.eig` | Exhaustive stimulus design against a model dir: enumerates every H/T pair over `--lengths` and greedily selects the max-joint-EIG set of `--select` stimuli (prior-predictive; `--registry` weights the models, `--responses` designs from the posterior; lazy batched greedy in float32 by default, `--no-lazy --scoring-dtype float64` for exact greedy). The pipeline's design stage runs this logic; the CLI is also usable directly. |
| `python -m src.critique.ppc` | The CriticAL posterior-predictive harness: computes agent-proposed test statistics on observed vs. replicated data (raw p + BH-FDR q). The inner loop runs it over the critique agent's statistics; usable standalone on any fitted model. |
| `python -m src.model_comparison.posterior` | Fit + ELPD-LOO-compare every model in a manifest dir on a responses CSV. |

**Cluster launchers (live runs — real money)** — see
`scripts/outer_loop_live/README.md` for the full runbook:

| Command | What it does |
|---|---|
| `bash scripts/outer_loop_live/run_pilot.sh` | Small single pilot from `pilot.yaml`: validates config + Prolific token, prints cost, confirms, submits one Slurm job. |
| `CONFIG=scripts/outer_loop_live/hero_run.yaml bash scripts/outer_loop_live/start_full_run.sh` | K parallel isolated runs of a full-scale config (`full_run.yaml` / `hero_run.yaml`). Live recruitment additionally requires `confirm_live_recruitment: true` in the yaml. |
| `sbatch scripts/outer_loop_live/setup.sbatch` | One-time cluster environment build (venv, functions deps, firebase CLI). |
| `python scripts/smoke_firebase_deploy.py` | Real Firebase deploy smoke: stages, deploys, drives simulated participants through the live page, and round-trips data via the token-guarded `/submit`/`/results` functions. `--confirm-production` gates it. |

**Validation harnesses (does the pipeline recover known ground truths?)** — see
`scripts/subjective_randomness/README.md`:

| Command | What it does |
|---|---|
| `python scripts/subjective_randomness/holdout_recovery.py` | Full agentic loop vs. a held-out registry ground truth; per-step recovery trajectory. |
| `python scripts/subjective_randomness/impossible_holdout_recovery.py` | Same, with deliberately-weird "impossible" ground truths outside every model family. |
| `python scripts/subjective_randomness/reanalyze_holdout_exhaustive.py` | Re-evaluate finished holdout runs on the exhaustive stimulus pool (no agents, cached fits). |
| `scripts/subjective_randomness/slurm/` | Sherlock submitters for the above (test–retest replicates + no-inner-loop ablations). |

**Inspecting results**

| Command | What it does |
|---|---|
| `python -m src.viewer.server` | Browser explorer for finished runs on disk (models, designs, critiques, posterior trajectories, transcripts). |
| `python -m src.viewer.freeze` | Freeze curated runs into a static site for public hosting. |
| `python -m src.monitor.server` | Live dashboard for an **in-progress** human study (Firestore + Prolific; flags degenerate data early). |

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

- `simulated_participants` (default): synthetic data from the experiment's
  PyMC model set (or a `--ground-truth-model`). No browser.
- `simulated_participants_nobrowser`: **LLM-as-participant** — each synthetic
  participant answers every stimulus directly via a language model. The jsPsych
  `3_implement` stage is skipped in a full run.
- `live`: real participants via Prolific + Firebase. Requires a deployed
  experiment to collect from (run with `--deploy-target firebase --prolific-mode
  live --confirm-live-recruitment`); it reads submissions from the token-guarded
  `/results` Cloud Function and refuses to fall back to synthetic data if no
  deployment is configured. Live deploys and collection both need
  `AUTO_PSYCH_RESULTS_TOKEN` in the environment (the shared secret for
  `/results`; see `docs/deployment_handoff.md`). The
  deploy puts the page up and checks the functions hold the token before it creates the
  Prolific draft (and, live only, publishes it), so a failed deploy leaves no
  study; it records the commit the code came from — from git, or from the
  `code_provenance.json` the live launchers write into their run copies — and
  refuses to deploy without one. Collection waits up to 3 hours for the
  study to fill, then pauses it and models what arrived.

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
implement stage (and inner-loop candidate + critique generation) is
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
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --coding-agent claude --claude-auth subscription
```

With Claude the run must say how its agents are billed: `--claude-auth
subscription` (`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`) or
`--claude-auth api` (`ANTHROPIC_API_KEY`), or `CLAUDE_AUTH` in the environment;
there is no default, and each agent gets only that mode's credential.

opencode runs headless via `opencode run`; grant it edit/bash permission in
`opencode.json` (the equivalent of Claude's `--dangerously-skip-permissions`),
and confirm its default model id (`google/gemini-3.1-pro-preview`) is valid for
your install (set a different model in `opencode.json` if not — the runners have
no model-override flag).

Project *assets* (problem definition, ground-truth models, featurizer) live under
`src/pipelines/outer_loop/projects/<project>/`. Generated experiment *outputs* are
written under:

```text
$AUTO_PSYCH_OUTPUT_DIR/<project>/experiment<N>/     (default: data/outer_loop/<project>/…)
```

**Set `AUTO_PSYCH_OUTPUT_DIR` to a directory outside the repository for any run
that collects data** (every stage, or `--agent 4_collect`). Collection keeps
every collected column — Prolific IDs, or in simulations the generating model's
name — in `<output>/<project>/raw_collected/`, and every agent can read the
repository (its working tree). `run.py` refuses to start such a run with the
default, in-repository output tree, before any stage runs. The live launchers
already use `$WORK_ROOT/<label>/data`.

The inner loop writes:

```text
model_loop/
model_loop/models/                             # the model zoo (seeds + admitted candidates, each with <name>.hypothesis.md)
model_loop/models/pruned/                      # models pruned as clear losers (audit trail)
model_loop/model_posterior.json                # posterior + ELPD-LOO + az.compare table (stacking weights)
model_loop/history.json                        # best model + posterior after every scoring step, and each round's critique status
model_loop/iter_<i>/critique/test_stats/*.py   # the critique agent's test statistics (none ⇒ the round had no critique)
model_loop/iter_<i>/critique/ppc_results.json  # empirical + FDR-adjusted p-values (absent when the round had no critique)
model_loop/iter_<i>/critique/critiques.md      # significant discrepancies of the incumbent (absent when the round had no critique)
cognitive_models/<winning_model>.py            # a NEW winning candidate, exported under its own name
cognitive_models/models_manifest.yaml          # the carried model set (name + hypothesis rationale)
token_usage.jsonl                              # one line per LLM call/agent run (tokens + cost, labelled by stage)
token_usage_summary.json                       # the experiment's total token spend, overall and per stage
```

A winning candidate is exported into `cognitive_models/` under the descriptive
name its agent chose; a seed that wins again is already in the set, so nothing
is copied. `model_registry.yaml` records a uniform prior over the carried set —
the model prior weighting the next experiment's EIG design (az.compare's
stacking weights remain a report field in `model_posterior.json`).

## Browse Run Results

A live web explorer presents one page per **run**. A run is any directory under
`data/` that holds experiments (or a single bare model loop) — for example
`data/outer_loop/subjective_randomness` or
`data/results/human_experiment/run1/subjective_randomness`.
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
      orchestrator.py        # seed/carry-forward, stage spawning, programmatic collect/deploy/inner-loop, validators
      collect.py             # collection modes (simulated / LLM-as-participant / live)
      participants.py        # closed (Gemini) + open (HuggingFace) participant models
      llm.py eig.py
      prompts/               # 3_implement / 4_collect_* prompts (no theorist or design stage)
      projects/<project>/    # problem_definition.md, ground_truth_models.py, evaluate_recovery.py, seed_models/
      deployment/            # firebase.py firestore.py prolific.py manifest.py local.py smoke.py
    inner_loop/
      run.py                 # `python -m src.pipelines.inner_loop.run` (entry point)
      pymc_orchestrator.py   # seed/score/critique/candidate rounds, novelty gate, pruning, export
      prompts/               # pymc_theory.md + critique.md
  critique/
    ppc.py                   # CriticAL posterior-predictive check (empirical p + BH-FDR)
  model_comparison/
    posterior.py             # ELPD-LOO softmax posterior over models + az.compare table (stacking)
    likelihood.py            # ELPD-LOO of one model
  models/
    pymc_inference.py        # load/fit (MCMC) agent-written PyMC models, PPC sampling, caching
    mcmc_defaults.py         # the ONE source of MCMC sampler defaults for every entry point
    theorist/                # loader.py + predictions.py (pure-Python prediction callables)
    project/                 # ground_truth.py
  subjective_randomness/     # research library: model families, recovery harnesses, stimulus pair pools
                             # (pymc_model_families/ = the frozen recovery registry — see its README)
  runtime/
    coding_agent.py          # backend-agnostic Claude Code / opencode subprocess launcher
    config.py console.py observability.py prolific.py
  registry/
    io.py                    # per-run model_registry.yaml (model -> weight, the EIG design prior)
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
uv run --group dev pytest -q                  # everything, including real-MCMC tests (~2 min)
uv run --group dev pytest -q -m "not slow"    # fast suite (~30 s) — skips MCMC sampling
```

# Auto-psych

Active development is organized around two explicit loops:

- `src/pipelines/outer_loop`: experiment loop. Claude Code agents propose models, design stimuli, implement experiments, collect data, then hand observed data to the inner model loop.
- `src/pipelines/inner_loop`: cognitive-model improvement loop. It fits and compares candidate models over a generic `Dataset`/`Trial` abstraction, exports the best model, and maintains model-zoo/BMC artifacts.

The old LangGraph/LangChain pipeline, Cloud Run entrypoint, SLURM submitter, and old root prompts have been retired into `legacy/`.

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
- `live`: real participants via Prolific (not yet wired into this runner).

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
`--hf-model` defaults to `Qwen/Qwen3.5-9B` (large — pass a smaller id for quick
runs).

**This participant model is separate from the coding-agent backend.** The
theory / design / implement stages (and inner-loop candidate generation) are
driven by `--coding-agent` (Claude Code by default); `--participant-backend` /
`--hf-model` only choose the model that *answers trials* during `4_collect`.

Smoke-test just the open participant path (no PyMC, no API key, tiny model):

```bash
uv sync --group open-models
uv run python scripts/smoke_open_participant.py
```

### Coding-agent backend

Both loops spawn a coding-agent CLI to write models/experiments. The default is
Claude Code; pass `--coding-agent opencode` (or set `CODING_AGENT=opencode`) to
use opencode instead. The backend is resolved once and exported so the inner
loop inherits it.

```bash
uv run python -m src.pipelines.outer_loop.run --project subjective_randomness --experiment 1 --coding-agent opencode
```

opencode runs headless via `opencode run`; grant it edit/bash permission in
`opencode.json` (the equivalent of Claude's `--dangerously-skip-permissions`),
and confirm its default model id (`anthropic/claude-sonnet-4-6`) is valid for
your install or override it with `--model`.

Project *assets* (problem definition, ground-truth models, featurizer) live under
`src/pipelines/outer_loop/projects/<project>/`. Generated experiment *outputs* are
written under:

```text
data/outer_loop/<project>/experiment<N>/
```

The inner loop writes:

```text
model_loop/
cognitive_models/inner_loop_model.py
cognitive_models/models_manifest.yaml
```

## Project Layout

```text
src/
  pipelines/
    outer_loop/
      run.py
      orchestrator.py
      prompts/
      projects/
      llm.py
      collect.py
    inner_loop/
      core.py
      likelihood.py
      fitting.py
      bmc.py
      zoo.py
      orchestrator.py
  runtime/
    config.py
    console.py
    observability.py
    prolific.py
  experiments/
    state.py
    state_loader.py
    problem_definition.py
    references.py
  registry/
    io.py
  models/
    theorist/
      loader.py
      predictions.py
    project/
      ground_truth.py
  validation/
    validators.py
legacy/
  run_pipeline.py
  run_agent.py
  run_critic.py
  prompts/
  src/
  tests/
  remote_jobs/
  cloudrun/
  batch_plots.py
  stats/
    correlations.py
  tests/
    test_correlations.py
```

## Tests

```bash
uv run --group dev pytest tests src/pipelines/inner_loop/tests -q
```

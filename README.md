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

# CONTEXT — experiment 1, agent 5_critique

**Project:** think_aloud_game24
**Experiment number:** 1
**Repo root:** /Users/ben/Documents/auto-psych
**This experiment directory:** /Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1

## Key paths

- Problem definition: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/problem_definition.md`
- Cognitive models dir: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/cognitive_models`
- Design dir: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/design`
- Experiment dir: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/experiment`
- Data dir: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/data`
- Responses: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/data/responses.csv`
- Model registry: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/model_registry.yaml`
- Critique output dir: `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/critique`

## All experiments (pooled data for posterior and PPCs)

Response files (all experiments, pass all to --responses):
- `/Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/data/responses.csv`

## Posterior command (run this exactly)

```bash
cd /Users/ben/Documents/auto-psych && python3 -m src.model_comparison.posterior \
    --responses \
        /Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/data/responses.csv \
    --models-dir /Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/cognitive_models \
    --out /Users/ben/Documents/auto-psych/cc_pipeline/projects/think_aloud_game24/experiment1/critique/model_posterior.json \
    --stimulus-col-a choices \
    --stimulus-col-b target \
    --response-col correct
```

## Column config (use these flags in all PPC commands too)

- stimulus_col_a: `choices`
- stimulus_col_b: `target`
- response_col: `correct`
- PPC flags: `--stimulus-col-a choices --stimulus-col-b target --response-col correct`

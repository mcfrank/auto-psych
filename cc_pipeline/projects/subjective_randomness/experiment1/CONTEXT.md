# CONTEXT — experiment 1, agent 5_critique

**Project:** subjective_randomness
**Experiment number:** 1
**Repo root:** /Users/ndg/auto-psych
**This experiment directory:** /Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1

## Key paths

- Problem definition: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/problem_definition.md`
- Cognitive models dir: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/cognitive_models`
- Design dir: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/design`
- Experiment dir: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/experiment`
- Data dir: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/data`
- Responses: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/data/responses.csv`
- Model registry: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/model_registry.yaml`
- Critique output dir: `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/critique`

## All experiments (pooled data for posterior and PPCs)

Response files (all experiments, pass all to --responses):
- `/Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/data/responses.csv`

## Posterior command (run this exactly)

```bash
cd /Users/ndg/auto-psych && python3 -m src.model_comparison.posterior \
    --responses \
        /Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/data/responses.csv \
    --models-dir /Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/cognitive_models \
    --out /Users/ndg/auto-psych/cc_pipeline/projects/subjective_randomness/experiment1/critique/model_posterior.json \
    --complexity-prior 0.1
```

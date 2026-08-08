# CONTEXT — experiment 1, agent 2_design

**Project:** subjective_randomness
**Experiment number:** 1
**Repo root:** /Users/ben/Documents/auto-psych
**This experiment directory:** /Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1

## Key paths

- Problem definition: `/Users/ben/Documents/auto-psych/src/pipelines/outer_loop/projects/subjective_randomness/problem_definition.md`
- Cognitive models dir: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/cognitive_models`
- Design dir: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/design`
- Experiment dir: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/experiment`
- Data dir: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/data`
- Responses: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/data/responses.csv`
- Model registry: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_registry.yaml`
- Inner model loop dir: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop`

## Current model set (the hypotheses your design must discriminate)

- **minkowski_accumulated_typicality**: People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, computing each event's typicality as a penalty based on its distance from a mental prototype, with the distance raised to a freely inferred Minkowski-like exponent so extreme feature deviations are disproportionately punished.
- **evidence_accumulation_messy_prototype**: Random-looking sequences are judged by an evidence-accumulation process where each item provides a baseline weight of evidence for randomness, discounted by the sequence's quadratic (and asymmetric, for alternation) deviation from a messy prototype — an ideal positive imbalance and ideal alternation rate — so near-ideal longer sequences are preferred while clearly deviant long sequences are penalized more heavily.
- **evidence_accumulation_per_run**: Random-looking sequences are judged by an evidence-accumulation process where each distinct run (streak of identical outcomes) — rather than each item — provides a baseline weight of evidence for randomness, discounted by the sequence's quadratic deviation from a messy prototype, so periodic patterns with artificially few runs are penalized without a separate periodicity cue.
- **artificial_balance_diagnosticity**: People judge randomness by Bayesian diagnosticity — the log-likelihood ratio of a fair coin versus a subjective "regular" generative process — where the regular process includes an "artificial balance" generator that targets an exactly equal count of heads and tails, explaining why people penalize suspiciously symmetric sequences as contrived rather than random.

Read each model's `.py` in the cognitive-models dir for its exact functional form, and `model_registry.yaml` for the current weight on each model (absent/empty registry = uniform).

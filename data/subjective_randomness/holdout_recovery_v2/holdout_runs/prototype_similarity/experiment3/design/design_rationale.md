# Design rationale — experiment 3 (prototype_similarity, holdout_recovery_v2)

## Overview

20 stimulus pairs selected from 177 candidates by EIG over a 5-model prior-predictive ensemble. EIG range: **0.2266 – 0.2843** (uniform model prior; model_registry.yaml has empty theories dict).

## Competing models

| Model | Key cue | New in exp3? |
|---|---|---|
| `fair_coin_run_baseline` | max_run excess vs kappa·log₂(n) | Yes |
| `encoding_compressibility` | max_run_norm + periodicity + imbalance | No |
| `bayesian_diagnosticity` | log-ratio fair vs {alt, streaky, biased} | No |
| `inner_loop_model` | Bayesian diag with learned streak_switch_prob | No |
| `alternation_prototype` | L1 distance of p_alts to learned theta_alt | No (0% posterior in exp2) |

## What the selected stimuli probe

The EIG-maximizing pairs converge on a single high-information contrast: **HHHHTTHH** (streaky + imbalanced, sequence A) vs **highly-alternating sequences** (sequence B). This directly discriminates the run-length models from the Bayesian-diagnosticity family:

- **HHHHTTHH** (n=8, max_run=4, h=6, p_alts≈0.29, imbalance=0.5): penalized by every model — high run excess (FCR), high max_run_norm + high imbalance (EC), streaky + biased (BD/inner), low p_alts far from prototype (AP).
- **Highly-alternating B sequences** (max_run=1–2, p_alts≈0.71–1.0): these split the models:
  - `fair_coin_run_baseline`: very low run excess (1 − log₂(n) ≤ −1) → looks **random**
  - `encoding_compressibility`: zero max_run_norm, but periodicity penalty may apply → partially prefers B
  - `bayesian_diagnosticity` / `inner_loop_model`: high p_alts is diagnostic of alternating generator → B looks **non-random**
  - `alternation_prototype`: p_alts≈1.0 is far from theta_alt (~0.65–0.75) → B looks non-random

The directional disagreement is sharpest for `fair_coin_run_baseline` vs `bayesian_diagnosticity`: FCR says the alternating sequence is more random (low run excess), BD says it is less random (alternating-generator match).

## Candidate pool construction

The 177 candidates were drawn from six targeted categories:
1. High-alternating vs balanced-moderate (FCR vs BD discrimination)
2. Cross-length same-log-excess pairs, low and mid levels (FCR vs EC length-normalization)
3. Same max_run, different imbalance (FCR indifferent; EC penalizes)
4. Periodic vs aperiodic with similar max_run (FCR indifferent; EC penalizes)
5. Medium-alternating vs fair-balanced (AP vs BD, carried from exp2)
6. Streaky vs alternating (EC/FCR vs BD)

The EIG naturally concentrated on category 6 (streaky vs alternating), confirming it as the strongest discriminating contrast under the current model ensemble.

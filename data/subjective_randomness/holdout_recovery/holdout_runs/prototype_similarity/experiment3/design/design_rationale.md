# Design Rationale — prototype_similarity Experiment 3

## Overview

- **Candidate pool**: 301 stimulus pairs (7 strategies)
- **Selected stimuli**: 20 pairs
- **EIG range**: 0.2153 – 0.2440
- **Models scored against**: 7 (5 inherited from Exp2 + 2 new Exp3 models)

## Model set

Experiment 3 inherits the 5 Exp2 models and adds two targeted additions:

| Model | Mechanism |
|-------|-----------|
| encoding_compressibility | Penalizes long runs, periodic templates, imbalance |
| bayesian_diagnosticity | Diagnostic likelihood vs alternating/biased generators |
| length_sensitive_prototype | Prototype model scaled by sqrt(n) |
| asymmetric_alternation_prototype | Prototype with asymmetric streak penalty |
| inner_loop_model | Statistical-expectation score (alts - 2ht/n) / n |
| **runs_test_model** (new) | Wald-Wolfowitz z-score: normalizes by sqrt(Var[R]) instead of n |
| **periodicity_salience** (new) | Template-matching: periodicity + imbalance, no alternation term |

## Candidate generation strategies

1. **Symmetric-deviation pairs** (60 sampled): streaky vs over-alternating at equal |p_alts - 0.5| and matched imbalance — discriminates `asymmetric_alternation_prototype` from `inner_loop_model`.

2. **Mixed-length pairs** (50 sampled): same alternation/imbalance profile at n=4 vs n=7–8 — discriminates `length_sensitive_prototype` from `inner_loop_model`.

3. **Compressibility pairs**: high-periodicity/long-run vs aperiodic at same imbalance — discriminates `encoding_compressibility` from prototype models.

4. **Bayesian pairs**: biased vs alternating sequences — discriminates `bayesian_diagnosticity` from prototype models.

5. **Classic pairs**: extreme streaky vs extreme alternating at each length — high information across all models.

6. **Imbalanced vs balanced at matched run count** (50 sampled, new): sequences with high imbalance (Var[R] → 0) paired against balanced sequences with similar alternation count — `runs_test_model` inflates z-score for the unbalanced sequence while `inner_loop_model` normalizes by n and treats them similarly.

7. **Periodicity-matched pairs** (50 sampled, new): high-periodicity sequences (periodicity ≥ 0.7) paired against low-periodicity sequences (≤ 0.4) with matched alternation rate and imbalance — `periodicity_salience` predicts the periodic sequence looks less random; alternation-based models predict ~50/50.

## EIG interpretation

EIG range 0.2153–0.2440 is consistent with previous experiments (Exp1: 0.32, Exp2: 0.32). The slightly lower EIG values reflect the flatter model posterior inherited from Exp2 (all 10 Exp2 models were statistically indistinguishable by ELPD-LOO), which reduces the information any single trial can contribute. The selected stimuli favor classic maximum-deviation pairs (all-H/all-T vs maximally-alternating), which produce the most reliable cross-model discrimination.

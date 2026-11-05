---
name: frequency-weighted-multiple-concepts
description: exp2 iter3 c2 — log-odds Bayes + k^-decay prior on multiples-of-k rules; MAP LL -2791.9 vs -2798.2 base
metadata:
  type: project
---
experiment2 iter_3 candidate_2 (2026-11-02), `frequency_weighted_multiple_concepts`: refinement of [[log-odds-distorted-concept-learning]], prior on mult-k ∝ k^-decay (decay≈1.7). MAP LL: base -2798.2; log-k decay -2791.9; linear excess over 12 -2797.4; step >12 -2797.6. Aimed at large_example_multiple_contrast critique.
**Why:** brief asked normative + one distortion; Bayes family still ~75+ nats behind exemplar models in-sample.
**How to apply:** familiarity priors on rules give only ~6 nats; Bayes gap lies elsewhere (gamma≈0.25 shows posterior far too extreme).

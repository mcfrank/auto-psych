# Design Rationale — Bayesian Diagnosticity Experiment 3

## Overview

- **Candidate pool**: 250 pairs generated from 45 curated sequences (lengths 4–8)
- **Selected stimuli**: 20 pairs (EIG range: 0.100 – 0.200)
- **EIG method**: prior-predictive sampling over 7 competing cognitive models

## Candidate generation strategy

Sequences were constructed to span four key axes of feature space, chosen to
create disagreements across the 7 competing models:

1. **Alternation rate** (`p_alts`): 0.14 – 1.0, including perfect alternation
   (HTHTHTHT) and very low alternation (HHHHTTTT).

2. **Max run length** (`max_run_norm`): 0.12 – 0.75, targeting the zone where
   `run_length_prototype` disagrees with alternation-based models (the former
   prefers intermediate runs ~0.3; the latter prefer short runs).

3. **Sequence length** (4–8): Cross-length pairs isolate the three
   length-sensitive models (`length_sensitive_alternation`,
   `length_sensitive_2d_prototype`, `bayesian_markov_fairness`) from the two
   proportion-scale models (`inner_loop_model`, `prototype_similarity`).

4. **Periodicity**: Periodic sequences (HTHTHTHT, HHTTHHTT) receive an extra
   penalty only in `encoding_compressibility`; non-periodic sequences with
   similar alternation rates expose that contrast.

## Selected stimuli — design themes

The 20 highest-EIG pairs share a common structural theme: one sequence has
a **single large deviation** in one feature dimension (e.g., very long runs
with "HHHHTTTT") while the paired sequence has **smaller, mixed deviations**
(e.g., "HHTH" is unbalanced and has a moderate run, but neither as extreme).

This geometry creates reliable disagreements because:
- `run_length_prototype` cares only about max run norm; it sharply distinguishes
  "HHHHTTTT" (max_run_norm=0.50) from sequences with intermediate runs.
- `encoding_compressibility` compounds run, periodicity, and imbalance penalties;
  it may rank a sequence differently than models that weight only one dimension.
- Length-sensitive models (`length_sensitive_alternation`, `bayesian_markov_fairness`,
  `length_sensitive_2d_prototype`) score cross-length pairs with identical
  proportions (e.g., "HHHT" n=4 vs "HHHHTTTT" n=8) differently from
  proportion-scale models, producing the highest EIG in the selected set.
- The top pair ("HHHT" vs "HHHHTTTT", EIG=0.200) contrasts a length-4 unbalanced
  sequence with a length-8 fully-balanced but extremely streaky sequence — a
  combination that forces disagreement on both the length and run-structure axes
  simultaneously.

## Self-check

- [x] `design/stimuli.json` exists, contains 20 JSON objects
- [x] All stimuli carry `sequence_a`, `sequence_b`, `eig` (numeric)
- [x] All EIG values > 0 (minimum 0.100, maximum 0.200)
- [x] Stimulus count (20) is within the 10–30 target range
- [x] This rationale file is non-empty

# Design Rationale — Bayesian Diagnosticity Experiment 2

## Overview

- **Candidate pool**: 250 pairs (generated from 132 canonical sequences, lengths 4–8)
- **Selected stimuli**: 20 (EIG top-20)
- **EIG range**: 0.078 – 0.212

## Five competing models

| Model | Features used | Core claim |
|---|---|---|
| `prototype_similarity` | imbalance, p_alts | Similarity to balanced, near-ideal-alternation prototype (linear) |
| `encoding_compressibility` | max_run_norm, periodicity, imbalance | Penalty for structural regularity |
| `inner_loop_model` | p_alts only | Quadratic distance from alternation-rate prototype |
| `length_sensitive_alternation` | alts, n | Count-scale alternation deviation (same count deviation = same penalty regardless of length) |
| `bayesian_markov_fairness` | alts, n | Log-Bayes-factor: transitions vs. fair Markov chain |

## Candidate generation strategies

Four complementary strategies were used to populate the 250-pair pool:

**S1 — Cross-length LSA vs ILM disagreement (up to 100 pairs)**
Pairs where the count-scale deviation (LSA) and proportion-scale deviation (ILM) rank the two sequences in opposite order. This exploits the mathematical fact that for sequences of different lengths, a sequence can be near-ideal on the count scale but off-ideal on the proportion scale (and vice versa). Requires |Δcount_dev| > 0.4 and |Δprop_dev| > 0.003.

**S2 — Imbalance manipulation (up to 70 pairs)**
Same- or cross-length pairs where imbalance differs by > 0.30 but p_alts is within 0.20. Targets the distinction between `prototype_similarity` (which penalises H/T imbalance) and `inner_loop_model` (which ignores it entirely).

**S3 — Structural regularity (up to 80 pairs)**
Pairs differing by > 0.40 in periodicity or max_run_norm while holding p_alts within 0.25. Targets `encoding_compressibility`'s unique sensitivity to run length and periodic structure.

**S4 — BMF vs LSA same-length pairs**
Pairs where one sequence sits near the fair-coin alternation ideal (alts ≈ (n−1)/2) and the other near the LSA prototype ideal (alts ≈ 0.65·(n−1)).

## Pattern in selected stimuli

EIG scoring ranked cross-length pairs highest (18 of 20 top stimuli span different lengths). The dominant pattern pairs a streaky sequence (low p_alts, 0.14–0.33) of one length with a slightly-less-streaky sequence of a different length. At these sub-ideal alternation rates, count deviation and proportion deviation give opposite verdicts, maximally discriminating the two families of count-based models (LSA, BMF) from the proportion-based models (ILM, PS).

Two same-length pairs appear in the set: HHTH/HTTHHTTH and HHHTH/HTHTTHHH. These contrast sequences at near-ideal alternation rates across different imbalance profiles, targeting the ILM vs PS and ILM vs encoding boundaries.

## Model discriminability

The design targets three key model boundaries simultaneously:
1. **Count vs proportion scale** (LSA / BMF vs ILM / PS) — via cross-length pairs with opposing count/proportion rankings
2. **Imbalance sensitivity** (PS vs ILM) — via same-proportion, different-imbalance contrasts
3. **Structural penalty** (encoding_compressibility vs all) — via high-run / high-periodicity pairs with comparable alternation rates

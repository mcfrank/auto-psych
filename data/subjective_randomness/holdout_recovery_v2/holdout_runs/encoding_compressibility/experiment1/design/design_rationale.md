# Design rationale — encoding_compressibility / experiment 1

## Models being discriminated

- **prototype_similarity**: scores sequences by distance from a balanced,
  moderately-alternating prototype. Uses `imbalance` and `p_alts`; the
  "most random" sequence is the one closest to (imbalance=0, p_alts≈θ_alt≈0.65).

- **bayesian_diagnosticity**: scores sequences by comparing their log-probability
  under a fair-coin process against the best-fitting structured alternative
  (alternating, streaky, or biased). Uses `n`, `h`, and `alts`; the "most random"
  sequence is the one least consistent with any of these structured templates.

## Key discriminating axis

The models disagree most on **high-alternation vs. low-alternation balanced
sequences**:

- `prototype_similarity` favours moderate-to-high alternation (p_alts near 0.65)
  because that is close to its prototype θ_alt.
- `bayesian_diagnosticity` penalises high alternation because such sequences
  strongly resemble an alternating Markov process (switch_prob = 0.95). It
  instead prefers sequences with p_alts ≈ 0.5 (pure independence under the fair
  coin).

A pair like HTHTHTHT (perfectly alternating, balanced) vs. HHHHHHHT (extremely
streaky, nearly balanced) illustrates the tension: both models would penalise
the streaky sequence, but they disagree on how much the alternating sequence
looks "random."

## Candidate generation

From all unique feature profiles among H/T sequences of length 4–8 (101
profiles, 4 753 pairs), we selected pairs where the two models predicted
**opposite relative orderings** (bayesian_diff × proto_diff < 0) and sorted by
disagreement magnitude. 150 disagreement pairs and 50 agreement pairs (for
contrast) were scored by EIG.

## EIG results

- **N stimuli selected**: 20 (out of 200 candidates)
- **EIG range**: 0.2311 – 0.2632 (all > 0)
- All top stimuli contrast a highly-alternating sequence (HTHTHTHT or HTHTHT)
  against a strongly-streaky or biased sequence (HHHHHHHT, HTTTTTTT, etc.)
  — exactly the axis on which the two models are predicted to disagree.

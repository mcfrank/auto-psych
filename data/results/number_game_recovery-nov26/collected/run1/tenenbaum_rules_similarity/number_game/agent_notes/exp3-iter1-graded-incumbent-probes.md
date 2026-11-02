---
name: exp3-iter1-graded-incumbent-probes
description: Exp3 iter_1 cand_4 MAP probes on graded_prior_digit_sum_concepts (LL -3513.54) — round edges +0.67 (submitted); primes +1.3 taken by cand_3; first-example weight, gain, digit product/difference, graded multiples/powers, range location all dead
metadata:
  type: project
---
iter_1 candidate_4 (2026-11-01, incumbent refinement slot). Base MAP LL -3513.54 (dsum_decay -0.25, scale 79, centre weight 0.017; MAP takes about 15 s).
- Primes own family +1.3: sibling candidate_3 had already submitted exactly this as graded_digit_sums_primes_family. **Check the sibling candidate dirs before submitting a graft.**
- Round-number range edges: -3512.87 (+0.67), bonus -0.46. Submitted as graded_digit_sums_round_edge_ranges (candidate_5 grafted round edges onto the flat digit-sum base instead).
- Logit gain + criterion: +0.55, gain 1.03 (dead). First-example weight: finds a local optimum with w 1.02 and a worse LL (dead).
- Digit-product family and digit-difference family: weight 0. Graded multiples prior and graded powers-by-base prior: +0.13 / +0.16. Linear or quadratic range-location prior: +0.18 / +0.43. All dead.
- The full check_candidate smoke run takes more than 10 minutes on this base, so run it in the background.
Probe scripts are in /tmp/c4i1 (map.py, mk*.py). The rmse.py script there gave implausible RMSEs (0.23), so it is buggy; don't trust it.
**How to apply:** this base is saturated; see [[exp3-iter1-graded-dsum-refinements]], [[exp3-concept-family-probes]], [[exp3-iter0-centre-weighted-probes]].

---
name: exp2-iter4-round-number-refinements
description: iter_4 probes on round_number_range_boundaries (MAP -2244.54): split lo/hi bonus, threshold round bonus, set-size exponent dead; primes +0.76, digit sums +0.60
metadata:
  type: project
---
iter_4 candidate_4 (2026-11-01), base round_number_range_boundaries MAP LL -2244.54 (bonus -0.64, scale 71):
- Separate lower/upper round bonuses (-0.52/-0.30): no gain. Round-cutoff bonus for thresholds: -0.07, dead.
- Set-size exponent k^beta on top of without-replacement: beta -0.003, dead (WR sampling absorbs it).
- Primes own family: -2243.78 (+0.76) — sibling candidate_3 took it (round_number_ranges_primes_family).
- Digit-sum family: -2243.94 (+0.60), data RMSE 0.0024 vs without_replacement_digit_sum_concepts (borderline novelty), 0.0059 vs base. Submitted as round_number_ranges_digit_sums.
Probe scripts /tmp/c4i4 (map.py: use model.value_vars names for compile_logp; rmse.py). May not persist.
**How to apply:** check sibling candidate dirs in the same iter before picking a graft; see [[exp2-memory-and-prior-anchoring-probes]], [[exp2-map-probe-results]].

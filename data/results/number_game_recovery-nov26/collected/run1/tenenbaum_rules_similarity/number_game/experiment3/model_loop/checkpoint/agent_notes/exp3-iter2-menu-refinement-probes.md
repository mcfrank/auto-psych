---
name: exp3-iter2-menu-refinement-probes
description: Exp3 iter_2 cand_5 MAP probes on centre_weighted_ranges_digit_sums (-3515.03) — range-only size exponent and Weber width dead; multiples-of-5 edge bias +0.5 over round edges (submitted)
metadata:
  type: project
---
iter_2 candidate_5 (2026-11-01, refinement slot). Base centre_weighted_ranges_digit_sums MAP LL -3515.03; round_edge_ranges_digit_sums -3514.03 (bonus -0.55).
- Range-only size-principle exponent g (density*(W/100)^(1-g)), exact computation without Chebyshev: g -> 1.00 when unbounded, dead. Exact version costs ~70 s per MAP.
- Weber-scaled width (W / (mid/50)^w in the Erlang prior): w -0.10, +0, dead.
- Round edges split into multiples of 10 (b10 -0.50) and multiples of 5 not 10 (b5 +0.44): -3513.51 (+0.52 over round), RMSE vs round 0.0048. Submitted as half_decade_round_edge_ranges.
- "Exclusive" decade edges (lo=x1, hi=x9 favoured, b_ex +0.29; on-ten b_in -0.67): -3513.69. As one parameter it ties round.
- The worst per-set misfits (69,9,39,21; 43,4,91,9; 83,38,58,68) look like binomial noise with about 5 responses per cell.
Probe scripts: iter_2/candidate_5/scratch/p (map.py, r5.py, rex.py, rss.py, weber.py).
**How to apply:** treat exp3 range-shape and size-principle tweaks as exhausted. See [[exp3-iter1-refinement-probes]], [[exp3-iter1-saturation-probes]].

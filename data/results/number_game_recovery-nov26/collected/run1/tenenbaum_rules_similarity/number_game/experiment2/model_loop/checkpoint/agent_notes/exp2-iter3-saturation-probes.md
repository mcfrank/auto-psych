---
name: exp2-iter3-saturation-probes
description: iter_3 probes on without_replacement_size_principle (MAP -2245.12): new families and sampling tweaks all collapse; graded Gaussian prototypes lose 70 nats
metadata:
  type: project
---
iter_3 candidate_0 (2026-10-30), each added alone to without_replacement_size_principle (MAP LL -2245.12; note the "contains digit" and exception families already sit at weight 0):
- "divisors of N" family: weight 0. Units-digit ranges ("ends in 7-9"): weight 0. Rules capped at an upper bound ("even up to 60"): weight 0.
- Family-first two-stage inference (softmax sharpness on family marginals): sharpness 0.988, data RMSE 0.0019 (near-duplicate).
- Spread/representative sampling (coverage^beta, normalised by the Beta(k-1,2) expectation): beta 0.004 (dead).
- Gaussian prototype magnitude concepts in place of sharp ranges (graded membership): -2315, i.e. 70 nats worse. Sharp range edges matter. Submitted as gaussian_prototype_magnitude_concepts (exploratory, expected to be pruned).
Residuals by target property (round, prime, repdigit, member, below/above, set size, digit reversal, sums/differences of examples) are all |z|<2.
Probe scripts: /tmp/i3c0/probe.py (MAP LL + RMSE for two model files), res.py (residual table); may not persist.
**Why:** the concept space and the sampling assumptions are saturated on exp2 data.
**How to apply:** don't spend another round adding concept families or one-parameter sampling tweaks; see [[exp2-map-probe-results]] and [[exp2-pedagogical-sampling-probe]].

iter_4 candidate_5 (2026-11-01, free refinement slot): on without_replacement_primes_family (MAP -2244.25), set-size amplification k^beta goes to beta -0.003 (no gain: without-replacement sampling and set-size amplification explain the same signal, don't stack them). Adding the digit-sum family instead gives MAP -2243.66 (incumbent round_number_range_boundaries -2244.54); data-row RMSE 0.006 vs WR-primes, 0.0096 vs WR-digit-sum, 0.011 vs incumbent. Smoke check OK in 96 s. Submitted as without_replacement_primes_digit_sums. Probe scripts /tmp/c5i4/map.py, rmse.py (may not persist).

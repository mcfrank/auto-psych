---
name: compressed-number-line-ranges
description: Compressed (power/log) mental number line for interval concepts in the number game — data want it nearly linear (c≈0.94)
metadata:
  type: project
---
iter_4/candidate_1 (2026-10-30), compressed_number_line_ranges: this is the incumbent, but interval examples are sampled evenly on the line x^c (log at c=0), p(x|[a,b]) = w(x)/L. With the reliability Beta-marginalised, the interval evidence is sum_j coef_j(u,v,n) * sum_intervals e_j(w)/L^j, where e_j is the elementary symmetric polynomial. prepare_observed tabulates log sums per set and per cell on an 81-point c grid in [0,1.6], and the graph uses Gaussian interpolation. prepare_observed takes 24 s.
Profile of observed loglik over c: 0.02 → -837.3, 0.3 → -829.2, 0.6 → -822.2, 0.94 → -819.0 (the mode), 1.0 → -819.0, 1.2 → -820.3. So the log line is ruled out by about 18 nats. MAP pool RMSE vs the incumbent is 0.004, data RMSE 0.003, and there is no gain.
**Why:** adults are linear on 1-100. The novelty margin is thin, since it only passes the 0.002 threshold because of the prior pull.
**How to apply:** don't retry magnitude-compression variants. See [[grid-tabulated-range-evidence]], [[number-game-refinement-findings]].

---
name: log-compressed-size-principle
description: exp2 iter_3 cand_2 submitted compressed_number_line_size_principle (log(x+c) perceived range size in likelihood, grid-interpolated c)
metadata:
  type: project
---
2026-10-19, exp2 iter_3 candidate_2 (exploratory, "normative + one distortion" brief): `compressed_number_line_size_principle`. It uses the no-conj base (kind prior, Erlang+decade, primes without 1, no upper edge). The range size in the size principle is its share of log(x+c). c is on the grid {inf,300,100,30,10,3,1}, picked by a Beta `compression` with linear interpolation of the precomputed group evidence and membership. This differs from the pruned log_scale_erlang_interval_prior, which compressed the prior and not the likelihood.
Check passed: smoke LL -2058.3, about 67 s per sampling run, whole check 3 min. At the initial point, moving compression changed p_yes by an RMSE of about 0.2, so it should pass the novelty gate unless the posterior puts compression near 0.
**Why:** Records which distortions have been tried. **How to apply:** Check its outcome before you retry magnitude-compression ideas. Related: [[concept-space-saturated]], [[pymc-model-speed-patterns]].

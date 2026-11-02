---
name: exp2-iter1-range-location-prior
description: Exp2 iter-1 cand_4 — range lower-bound location prior +1.8 nats MAP over incumbent; fuzzy edges/lambda heterogeneity null; Taylor-moment speed trick (25x)
metadata:
  type: project
---
Exp2 iter 1, candidate_4 (2026-10-30), refining narrow_range_specificity_bias. The incumbent's MAP NLL is 1927.33.
One-parameter screens:
- range log prior + beta*(lo-1)/99: 1925.52 (beta 0.65, and gamma collapses to -0.02).
- hi bound instead of lo: 1926.0.
- log-midpoint prior: 1925.75.
- relative width: 1925.84.
- Gauss-Hermite heterogeneity of the rule/range log-odds: 1926.87.
- fuzzy range edges: exp kernel 1927.32, relative to width 1926.85, downward only 1927.29.

The lo-prior model's RMSE is 0.005 against relative_width and 0.013 against the incumbent. Submitted `range_location_gradient_prior`; smoke loglik -1928.4.

Speed trick for any per-interval prior that is not a function of width alone: expand exp(beta*u) as a Taylor series (24 terms) within each width group, with u the offset from the width's centre. Precompute the moments per set and per unique (set, target) cell; exp2 has only 895 cells. The model is then two matvecs with coefficients that don't depend on the set. This is exact to 1e-13, takes 2.3 ms per gradient versus 51 ms dense, and the smoke check runs in 30 s. Gathering post[cell_set] or using a cumsum lo/hi grid was slower than dense.

**How to apply:** fuzzy edges are dead on exp2 as well (cf. [[iter3-screens]]). See [[exp2-iter0-relative-width-prior]] and [[fit-speed-budget]].

---
name: fuzzy-and-centre-weighted-ranges
description: exp2 iter_3: any non-uniform within-range generator or graded range readout loses; uniform sharp intervals optimal; subset pre-sum speed trick
metadata:
  type: project
---
Exp2 iter_3 cand_0 (2026-11-01). Base: familiar_strays_family_level_prior (MAP ll −2031 in my prof script). Swapped in variants of the interval part:
- Gaussian clusters (log-σ grid, integer μ, typicality readout): −2093. Box readout at 1.5σ: −2084. Generalised-normal β 4–16: −2079 (the grid and log-σ prior cost ~45 nats).
- One cluster per interval (μ = midpoint, σ = w/√12), uniform prior. Fuzzy generation + typicality readout: −2114. Gaussian generation + sharp membership: −2047. Uniform generation + typicality readout: −2098.
- Truncated within-range generation, σ = w/√12, w/4, w/2: −2047, −2058, −2033. Edge-weighted (1/kernel), σ = w/√12 and w/2: −2047, −2033. Uniform is the optimum in both directions.
Submitted centre_weighted_range_sampling (σ = w/√12, −2047 MAP, smoke OK).
**Why:** the data want sharp, uniform ranges. Graded readouts are the worst part.
**How to apply:** don't propose fuzzy or typicality-graded magnitude concepts again. Speed trick: for any parameter-free per-example generator g_h, pre-sum Π_{i∈J} g_h(x_i) over h per subset J (16 per set, plus target-weighted per cell). Evidence = Σ_J c_|J| W_J G_J. That gives 2.9 ms/grad over all 5050 ranges, against 240 ms for the explicit (sets, H, 4) form.
Related: [[compressed-number-line-ranges]], [[grid-tabulated-range-evidence]], [[graded-exemplar-uncertain-reliability]]

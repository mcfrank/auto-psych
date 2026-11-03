---
name: graded-neighbourhood-magnitude
description: exp2 iter_1 cand 0: replacing hard-edged ranges with graded similarity neighbourhoods (centre, breadth) loses ~120 nats at smoke fit
metadata:
  type: project
---
experiment2 iter_1/candidate_0 (2026-10-30), gaussian_neighbourhood_magnitude_concepts: incumbent rules unchanged, intervals replaced by neighbourhoods (centre c, breadth w on a 24-value log grid, lognormal breadth prior, examples sampled ∝ similarity, readout = graded similarity). Smoke ll: exponential kernel -2398, Gaussian -2391, vs incumbent ≈ -2273. Fit ~10 s per 200 draws.
**Why:** people's yes-rate is flat inside the examples' span (interior_proximity_gradient null, [[experiment2-iter1-critique-stats]]); graded gradients can't produce that plateau. Fits with fuzzy-edge null in [[round4-null-refinements]].
**How to apply:** similarity-brief slots should keep hard-edged ranges; put the similarity metric elsewhere (e.g. digit/parity), not on magnitude.

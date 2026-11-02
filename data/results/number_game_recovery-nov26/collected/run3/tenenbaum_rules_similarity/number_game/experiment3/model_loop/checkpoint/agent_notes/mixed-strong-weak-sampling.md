---
name: mixed-strong-weak-sampling
description: exp2 iter_3 c4: Navarro mixed strong/weak sampling on beta_share incumbent — theta≈0.90±0.07, LOO +0.1, p_yes RMSE 0.0056 (novelty risk)
metadata:
  type: project
---
exp2 iter_3/candidate_4 (2026-10-30), submitted as mixed_strong_weak_sampling_concepts. P(x|h) = 0.9·[x∈h]·(θ/|h| + (1−θ)/100) + 0.001, θ~Beta(4,1). Intervals grouped by (size, #examples inside, k≤4) in prepare_observed. At θ=1 it reproduces the incumbent exactly.
Quick 2x300 NUTS: theta 0.899±0.065, LOO −2273.49 vs incumbent −2273.57 (null). p_yes RMSE vs incumbent 0.0056, max 0.016, so it may fail novelty. The smoke check takes about 3 minutes.
**Why:** the size principle's strength is well calibrated; this joins evidence-exponent and without-replacement as null sampling tweaks ([[experiment2-round0-readout-and-sampling]], [[experiment2-null-refinements]]).
**How to apply:** don't retry sampling-assumption variants on exp2.
Pitfall: pytensor compiled functions reuse output buffers, so copy each result. Also compile_fn(ip) gave inconsistent p_yes, so use graph_replace + pytensor.function.

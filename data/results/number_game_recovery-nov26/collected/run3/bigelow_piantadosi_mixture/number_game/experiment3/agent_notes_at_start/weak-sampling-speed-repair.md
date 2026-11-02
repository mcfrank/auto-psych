---
name: weak-sampling-speed-repair
description: exp2 weak_sampling_concepts was too slow; fixed by family-only rule weights + contracting size axis with pt.dot first
metadata:
  type: project
---
exp2 iter_0 cand 1: weak-sampling (no size principle) model timed out with 45 per-rule deviations + tau. Repair: one weight per rule family (parity fixed 0), and contract (cells,k,size) count tables with the shared size prior via pt.dot before gathering k-likelihoods → 12 ms → 2.5 ms per gradient; 2 chains×2000 iters in 3 min, ~24 leapfrog steps, r_hat 1.0. Posterior: eps≈0.06, width prior beta≈1.17, lambda_0≈0.7. Smoke ll -2225 (vs incumbent ~-2056 LOO), so likely loses — that is the informative outcome.
Related: [[grid-tabulated-range-evidence]], [[interval-tempering-speed-trick]]

---
name: grid-tabulated-range-evidence
description: When one scalar parameter enters the 5050-interval tables non-trivially, tabulate per-set/per-cell interval aggregates on a grid in prepare_observed and interpolate smoothly
metadata:
  type: project
---
iter_3/candidate_1 (exemplar_retrieved_number_concepts, 2026-10-30): magnitude-similarity kernel r^|x-y| inside the interval likelihood gave 0.13-0.17 s/grad with full 5050x100 tables. Fix: the intervals depend only on r, so prepare_observed tabulates log summed interval evidence per set and the endorsement share per cell on 221 logit(r) points; the graph interpolates. That made it 1.5 ms/grad.
**Why:** linear (tent) interpolation made warm_start_nuts fail with "Hessian not PD" because of the kinks; normalised Gaussian-kernel weights (width = grid step) fixed it.
**How to apply:** any single-scalar change to the interval part that the (n,k,s) trick in [[interval-tempering-speed-trick]] can't handle. Outcome: the mode was at r≈0.016 and the smoke loglik was -831.6. The novelty-check result was still pending when this was written.

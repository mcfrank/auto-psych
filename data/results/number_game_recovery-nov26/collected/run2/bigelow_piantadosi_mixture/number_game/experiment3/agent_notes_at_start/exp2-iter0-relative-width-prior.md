---
name: exp2-iter0-relative-width-prior
description: Exp2 iter-0 cand_4 repair — Weber relative-width range prior (width/midpoint)^-k +2.7 nats MAP, distinct from absolute-width prior; runs/digit-share/gain fail
metadata:
  type: project
---
Exp2 iter 0, candidate_4_repair_1 (2026-10-30). The first attempt (absolute width prior) was rejected as a near-duplicate of narrow_range_specificity_bias (pool RMSE 0.0014).
MAP screens on the WOR incumbent (NLL 1928.49):
- range prior (width/midpoint)^-k: 1925.84 (k=0.18), RMSE 0.013 vs the absolute-width model. Best so far. (width/hi)^-k gives 1926.1, and a midpoint-only prior gives 1927.8.
- WOR twos-runs family: its share goes to -inf (log share -4 costs 37 nats), so it's ruled out.
- digit-family share: sharply pinned at 0, a near-duplicate.
- log-odds gain: 0.976, +0.2 nats, RMSE 0.003. Too close to the incumbent.
Submitted `relative_width_range_prior`. It bins log(width/mid) at 0.05 and takes a logsumexp over the bins. It takes 10 s per 200 iterations, and the smoke loglik is -1928.6.

**How to apply:** magnitude-relative range priors are the live direction on exp2. See [[exp2-iter0-narrow-range-prior]].

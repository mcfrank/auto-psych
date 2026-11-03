---
name: exp2-iter0-narrow-range-prior
description: Exp2 iter-0 cand_4 — MAP screens on WOR incumbent (NLL 1928.5); guess_yes→1; interval width prior size^-k +1.2 nats submitted
metadata:
  type: project
---
Exp2 iter 0, candidate_4 (2026-10-30). MAP on the incumbent: NLL 1928.49, lapse 0.006, guess_yes→1 (logit 30). So the lapse is a pure "yes" floor.
Screens, each with one extra param: free outlier rate 1928.40; size-principle exponent 1928.49; digit-family share 1928.48; Erlang interval prior 1931.3 (worse); interval prior ∝ width^-k 1927.33 (k=0.12, RMSE 0.010 vs the incumbent).
Submitted `narrow_range_prior_without_replacement`: the width prior plus a yes-only lapse, which keeps 3 params. Smoke loglik -1928.7.
Speed trick: group the intervals by width (100 groups). Precompute the log-evidence per (set, width) and the log member-mass per (unique cell, width); in the model, do a logsumexp over widths. That gives ~10 s per 200 iterations.

**Why:** calibration bins are already within ~0.01; exp2 is again at the ceiling.
**How to apply:** don't re-screen outlier/exponent/digit-share on exp2. See [[number-game-ceiling]], [[exp2-iter0-critique]].

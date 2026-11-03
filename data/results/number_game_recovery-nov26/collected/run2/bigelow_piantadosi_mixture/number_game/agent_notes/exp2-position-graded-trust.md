---
name: exp2-position-graded-trust
description: Exp2 iter-0 cand 2 — position-graded stray rate on WOR incumbent, +1.1 nats MAP, slope ≈ -0.15, submitted
metadata:
  type: project
---
Exp2 iter 0, candidate_2 (2026-10-30), brief asked for an order effect. I submitted `position_graded_example_trust`: the WOR incumbent, but the log-odds of a stray change linearly with an example's position, centred so the mean stays at 0.1. MAP NLL is 1936.83 vs 1937.94 at slope 0, so +1.1 nats, with slope ≈ -0.15 (a slight lean toward trusting later examples). Smoke loglik -1929.9, about 4 ms per logp+grad.
Implementation: per set, Chebyshev tabulation in the slope over [-3, 3] with 21 nodes (error about 1e-8). The interpolated quantities are the log-evidence and the in_concept rows, gathered per row. See [[iter3-process-screens]] and [[number-game-ceiling]].
**Why:** order effects are near the noise ceiling again.
**How to apply:** don't re-screen graded stray-by-position. It is distinct from order_weighted (likelihood powers) and first_example_anchored.

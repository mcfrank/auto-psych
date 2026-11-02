---
name: exp2-iter1-digit-sum-concepts
description: Exp2 iter-1 concept-family MAP screens on narrow-range incumbent; digit-sum family +2 nats, submitted digit_sum_concepts
metadata:
  type: project
---
Exp2 iter 1, candidate 2 (2026-10-30). The narrow-range incumbent's true MAP NLL is 1927.33, with lapse 0.009 and guess→1. The fit script's 1930.6 was a local minimum. I added each concept family on top of the incumbent with a free log-share relative to the rules. The near-miss family (rule ∪ one extra number, 3894 concepts), unions of two final digits, repdigits, "tens digit is d" and "tens or units digit is d" all went to -inf share, so none gained. Only "digits sum to s" (18 concepts) gained: NLL 1925.26 at log share ≈ -2.3. I submitted `digit_sum_concepts` with log_sum_share ~ N(-3, 1.5). Smoke loglik was -1928.4 and the check took 24 s per 200 iterations. The screener is at experiment2/model_loop/iter_1/candidate_2/scratch/screen2.py.
**Why:** this confirms [[iter3-concept-family-screens]] and [[number-game-ceiling]]: concept additions buy only about 2 nats.
**How to apply:** don't re-screen these families. Start MAP fits from lapse≈-4.7 logit, or the optimizer sticks at 1930.6.

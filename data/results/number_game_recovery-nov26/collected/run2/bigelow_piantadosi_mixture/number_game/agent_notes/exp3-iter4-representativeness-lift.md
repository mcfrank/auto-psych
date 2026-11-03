---
name: exp3-iter4-representativeness-lift
description: Exp3 iter-4 cand_2 — representativeness judgment (posterior logit minus w * base-rate logit) on the position-decaying incumbent, smoke -3192.9, submitted
metadata:
  type: project
---
Exp3 iter 4, candidate_2 (2026-10-31), exploratory slot. Submitted `representativeness_lift_judgment`: yes log-odds = posterior logit - w * prior-membership logit. The base rate is computed analytically under the same concept prior (rule mean, sum mean, t(101-t)/5050). It adds 1 parameter (w ~ N(0, 0.5)), 19 s per smoke run, smoke LL -3192.9 (incumbent about -3192.4). I ran no MAP screen, so I don't know whether w is null.
Raw 31,3,1 → 31,3,1,15 data: after adding 15, even targets within 1-31 rise (4: 0→0.8, 20: 0→0.8), with n=5 per cell, so it is noisy.
See [[exp3-iter4-critique]], [[exp3-iter3-position-decaying-evidence]].

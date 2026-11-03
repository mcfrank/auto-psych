---
name: wsls-exact-recursion
description: Exact, fast win-stay/lose-sample (order-dependent) concept model for the number game via prefix-posterior recursion
metadata:
  type: project
---
iter_2/candidate_0 (2026-10-30): win_stay_lose_sample_concepts. The guess distribution is tracked exactly: D_t = stay*c_t*D_{t-1} + R_t*P_t. Rules are tracked individually; with alpha fixed at 0.9, interval likelihoods are parameter-free, so all interval sums (prefix evidence, the share containing later examples, plus the target per cell) are precomputed. Sets are padded to 4 steps, with stay=1 and R=0 on dummy steps. Matches brute-force simulation to 1e-15; smoke fit took 15 s, loglik -831.6.
**Why:** the brief asked for an order effect; serial-position weighting had been rejected as a near-duplicate.
**How to apply:** reuse for any sequential/running-guess model. See [[interval-tempering-speed-trick]].

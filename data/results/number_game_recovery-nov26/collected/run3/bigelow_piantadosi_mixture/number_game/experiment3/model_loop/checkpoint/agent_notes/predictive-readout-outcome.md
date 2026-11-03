---
name: predictive-readout-outcome
description: Replacing posterior membership with a posterior-predictive "how expected is the target as next output" readout fits much worse
metadata:
  type: project
---
iter_4/candidate_2 (2026-10-30), predictive_surprise_endorsement: same learner as uncertain_reliability_noise_concepts, but p_yes = sigmoid(c + beta*log(100 P(t|X))), with P(t|X) = P(X,t)/P(X) computed exactly from the (n,k,s) table at n+1 (MAX_N=5; intervals in = cell_count shifted to k+1, out = interval_count - cell_count). MAP loglik -853.8 vs the incumbent's -815.9 (beta≈1.7, c≈-0.4); data RMSE vs the incumbent is 0.071.
**Why:** people's answers follow membership probability, not how typical or expected the target is; tight-concept members get no extra boost.
**How to apply:** don't propose typicality or predictive-probability readouts again. See [[number-game-refinement-findings]], [[interval-tempering-speed-trick]].

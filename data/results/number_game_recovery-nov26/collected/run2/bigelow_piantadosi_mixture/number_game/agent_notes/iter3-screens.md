---
name: iter3-screens
description: Round 3 MAP screens on rule_breadth_prior_concepts (NLL 903.3); sampling without replacement +0.5 nats with 0 params, submitted
metadata:
  type: project
---
Round 3, candidate_4 (2026-10-30). Baseline numpy MAP NLL was 903.3. Fuzzy interval edges gave 903.2. A log-quadratic breadth prior gave 902.8 (+1 param). Scaling gamma by the number of examples gave 903.3, and gamma divided by n gave 904.0. Strong sampling *without replacement* gave 902.8 with no new parameter. Its prediction RMSE against the incumbent is 0.0074 (largest change at 8,64 → 16), so it clears the novelty gate. I submitted it as `concepts_sampled_without_replacement`, with smoke loglik -904.3 and about 3 s per 200 iterations. The screener script is in iter_3/candidate_4/scratch/s/.

**Why:** this confirms [[number-game-ceiling]] again. Residual sets (14,47; 83,38,58,68; 85,19,91) look like binomial noise.
**How to apply:** don't re-screen these forms. Candidate_3 refined the same incumbent in parallel this round.

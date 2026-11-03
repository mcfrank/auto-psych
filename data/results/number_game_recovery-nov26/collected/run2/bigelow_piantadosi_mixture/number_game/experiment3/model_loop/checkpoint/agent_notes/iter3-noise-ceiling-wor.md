---
name: iter3-noise-ceiling-wor
description: Round 3 cand_5: incumbent is at the binomial noise ceiling; participants homogeneous; without-replacement sampling ties incumbent with 0 extra params
metadata:
  type: project
---
Round 3 candidate_5 (2026-10-30). The CSV has 5 participants (trial_index resets every 450 rows), and each one answers all 450 (set, target) cells once. Their yes rates run from 0.24 to 0.28.

MAP screens on `rule_breadth_prior_concepts` (NLL 903.26):
- per-person lambda_0: 902.5
- per-person gamma: 902.8
- per-person lapse + guess rate (10 extra params): 900.65
- per-person example noise (5 extra params): 900.9

So the data show no individual differences. The incumbent's NLL is 903.3, against a saturated per-cell NLL of 703.8. That excess of about 200 nats is below the roughly 225 expected from pure binomial noise, so the model is at the noise ceiling.

Submitted `without_replacement_size_principle`: the equal-prior lapse model, with each member example drawn without replacement (probability 1/(|h| - earlier member examples)). It adds no parameter. MAP NLL was 903.26, the same as the incumbent, which has one more parameter. Prediction RMSE against both models is about 0.013. Smoke loglik was -904.9.

**Why:** at the noise ceiling, only parsimony can win ELPD.
**How to apply:** don't screen participant-level structure again. See [[number-game-ceiling]] and [[iter2-screens]].

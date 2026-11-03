---
name: exp2-final-digit-rule-discount
description: Exp2 iter-0 cand_3 repair — width-prior dup rejected (RMSE 0.0004); "ends in d" family prior weight +0.2 nats, RMSE 0.004 vs incumbent, submitted
metadata:
  type: project
---
Exp2 iter 0, candidate_3_repair_1 (2026-10-30). The first attempt, an interval width power prior, was rejected as a near-duplicate of narrow_range_specificity_bias (RMSE 0.00044). Three other slots also submitted the width prior. That mechanism is now taken.
Screens on the WOR incumbent (NLL 1928.53): a log-weight on the "ends in d" family gives 1928.31 (w=-0.075). A weight on the "contains d" family gives 1928.38. At MAP, the ends-in model's RMSE is 0.004 vs the incumbent, 0.0057 vs rule breadth and 0.011 vs the width prior.
Submitted `final_digit_rule_discount`: one Normal(0,1) log-weight, with the rule softmax computed inside the model. The smoke loglik was -1930.9, and the check took 35 s.
**Why:** novelty is gated at an RMSE of 0.002, so family-level rule weights clear it while still tying.
**How to apply:** check novelty against all live models before submitting a prior tweak. See [[exp2-iter0-screens]] and [[number-game-ceiling]].

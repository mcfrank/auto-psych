---
name: exp3-iter4-generative-predictive-strength
description: Exp3 iter-4 cand_2 repair — representativeness lift was a novelty dup (w→0, RMSE 0.0007); replaced by a posterior-predictive "generation" judgment, MAP -3580 (~390 nats behind), submitted as a loud test
metadata:
  type: project
---
Exp3 iter 4, candidate_2 repair (2026-10-31). The first attempt, `representativeness_lift_judgment`, was rejected as a dup of position_decaying (RMSE 0.00069) because the base-rate discount w fit to about 0.
The repair is `generative_predictive_strength`: yes = sigmoid(slope*(log(100*pred) - criterion)), where pred = ALPHA*E_post[1{t∈h}/|h|] + STRAY. Interval entries carry entry_rate = count/width; folding 1/width into the per-entry counts cut a gradient from 19 to 8 ms, and the smoke run takes 17 s. MAP LL -3580 against about -3190 for the incumbent (lapse 0.13). A size-weighted generation reading of the question is strongly rejected; people judge membership.
**How to apply:** don't re-propose predictive-probability or likelihood-ratio-vs-chance response rules. Exp1's screen also gave about -40. See [[exp3-iter4-representativeness-lift]], [[iter4-process-screens]].

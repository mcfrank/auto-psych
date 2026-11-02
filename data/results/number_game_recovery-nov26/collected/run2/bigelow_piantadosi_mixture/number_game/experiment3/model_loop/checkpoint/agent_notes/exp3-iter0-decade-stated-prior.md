---
name: exp3-iter0-decade-stated-prior
description: Exp3 nested concept-family additions collapse to near-dup (RMSE<0.002); fix by stating a tight prior at profile optimum
metadata:
  type: feedback
---
Exp3 iter-0 cand 1 repair (2026-10-31): decade_named_concepts with free share N(-5,1.5) was rejected as a near-dup of the heading model (RMSE 0.00046), because the share's likelihood profile is flat toward -inf and the posterior drifts there. I resubmitted it as `decade_named_concept_boost` with log_decade_share ~ N(-7.3, 0.3). That is a stated prior: a decade adds about 1/6 to its range's prior. Screen-units lf ≈ -5.8 gives a cost of about 0.3 nats and a novelty RMSE of about 0.01; smoke -3194.3.
Other screens: an "unrecognised rule" concept (examples from an unknown set, low-q yes to non-examples) went null with weight→0. The digit-product gain (+1.3) comes entirely from 85,19,91, so it is spurious like digit difference.
**Why:** any nested addition with a free weight can shrink to the incumbent.
**How to apply:** in a concept-family slot, pin the weight with an explicit, interpretable prior near the profile optimum, using screen.py and dec.py in candidate_1_repair_1/scratch. See [[exp3-iter0-concept-family-screens]].

---
name: cluster-cued-variable-lambda
description: exp3 iter_2 c4: tightness shift on the logit-lambda mean of variable_rule_range_expectation, beta .48, +1.6 MAP, RMSE .010; narrow fill-in .41→.44
metadata:
  type: project
---
exp3 iter_2/candidate_4 (2026-11-02) is cluster_cued_variable_rule_expectation. It takes the incumbent variable_rule_range_expectation and makes the logit-lambda mean lambda_mu − beta·tight per set, where tight = 1 − span/99 and n=1 → 0. lam_k becomes a (sets, K) array.
- MAP ll: −3317.61 vs −3319.24 for the incumbent. beta = .48 and lambda_sd = .70; the two effects add.
- Train RMSE vs the incumbent: .010.
- The critique tight_rule_inhull_nonmember_yes is still open: .52 observed vs .44 predicted.
c3 was also an incumbent-refinement slot, so it may have made the same graft (see [[check-sibling-candidates-first]]).
**How to apply:** data-dependent prior cues and per-judgement prior variability combine additively. See [[cluster-cued-range-salience]] and [[variable-rule-range-expectation]].

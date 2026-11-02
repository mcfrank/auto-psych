---
name: forward-continuation-response-bias
description: exp3 iter3 decision-stage screens on round_landmark_veto — signed above/below logit tilt +3.7 loglik; gain/criterion/any-k/all-k and format veto null
metadata:
  type: project
---
2026-10-31, exp3 iter_3/candidate_2, MAP screens on round_landmark_veto (base MAP loglik -3415.3):
- Signed logit tilt, +1 if target is above the largest example and -1 if below the smallest: -3411.6 (tilt 0.14). A two-sided version gives up≈down≈0.14, so the effect is symmetric. Submitted as `forward_continuation_response_bias`.
- Null (within 0.2): logit gain+criterion (gain 1.03), 1-(1-p)^k, p^k, and a numeral-format veto (single digit / round / other). The single-digit part fits to 0, and adding it makes the round veto worse (-3418.7).
- The base is well calibrated by predicted bin and on examples-as-targets (obs 0.757 vs pred 0.791). Raw residuals: above obs 0.387 vs pred 0.369; below 0.258 vs 0.278.

**Why:** on exp3 the decision stage is probability matching. Only the above/below direction leaves a residual.
**How to apply:** scripts are in iter_3/candidate_2/scratch (map.py, dump.py, dir_*.py). See [[exp3-decision-stage-screens]], [[forward-counting-tilt-finding]].

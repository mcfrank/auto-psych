---
name: single-example-breadth
description: Logit lean-to-yes on one-example sets gave +17.5 MAP on seed_multiple_digit_overshadowing (exp2); dropping bounds for n=1 +8.8; 1/n form +15
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_1/candidate_5, single_example_open_generalization). Full-data MAP log-lik: base seed_multiple_digit_overshadowing -3157.1, incumbent seed_overshadowing_last_digit_recency -3152.7.
- intercept + breadth*[n==1]: -3139.6 (breadth about 0.86). Submitted. Smoke log-lik -3146.2, fit 9 s.
- breadth/n: -3141.9. Magnitude bounds dropped for one-example sets (0 params): -3148.3.
- Shrinking overlap toward a prior share by n/(n+k): -3153.3. Slope times n/(n+k): 0 (k went to 0).
**Why:** the critique single_example_yes_gap (q=0.008). The base under-predicts singles (0.556 vs 0.588) and over-predicts 2-example sets.
**How to apply:** the incumbent has not had this tried yet; the recency graft plus breadth probably stack. See [[last-example-digit-recency]], [[overlap-model-tweaks]].

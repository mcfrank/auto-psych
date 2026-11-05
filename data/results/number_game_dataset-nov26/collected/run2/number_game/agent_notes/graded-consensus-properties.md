---
name: graded-consensus-properties
description: Letting partially shared properties count (weight = fraction^gamma) gave +3.2 MAP loglik on final_digit_overshadowing_overlap (exp2)
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_0/candidate_1, graded_consensus_feature_overlap). Experiment 2 has 5040 rows and 94 sets. The incumbent's MAP loglik is -3197.1; with every property held by k of n examples weighted (k/n)^gamma, it is -3193.8, with gamma about 2.5. The motivation was 13,91: targets ending in 1 (the last example's digit) get about 0.8 yes, other odd targets about 0.3, and the incumbent predicts about 0.6 for all odd targets. The incumbent also over-predicts 84,94,34 / 78,48,88 / 46,76,66 non-digit targets (0.17 predicted, about 0.03 observed), and under-predicts the 7,63 even multiples of 7 (it has no seed family).
**Why:** the gain is modest. A recency effect (the last example's digit) is the critique's lead and is still untested on exp2.
**How to apply:** next, try last-example weighting or a lower floor for overshadowed sets. See [[final-digit-overshadowing]], [[overlap-model-tweaks]].

---
name: leaky-memory-recency
description: Leaky memory for earlier examples (recency) on the overshadowing overlap model gave ~0 gain on experiment2 data
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_0/candidate_0, leaky_memory_example_elimination). Each earlier example eliminates the properties it lacks only with probability memory**lag. MAP memory was 0.993, and the MAP log-lik was the same as with memory fixed at 1 (-3197.5 vs -3197.1). The weak last-vs-first final-digit critique (q=0.4) is not explained by forgetting. In raw data, targets sharing the last example's digit get 0.66 yes and those sharing the first's get 0.56 (396 trials each).
**Why:** this is the third order/recency mechanism to fail (with recency_weighted_example_evidence and position_weighted_feature_tally).
**How to apply:** don't spend more effort on order effects. See [[final-digit-overshadowing]] and [[overlap-model-tweaks]].

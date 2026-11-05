---
name: shared-divisor-factor-exemplar
description: exp2 iter_1 c3 — penalty only if gcd(target, set-gcd)<3 (singles use example) on common_divisor_sharpened incumbent, +6.5 nats MAP
metadata:
  type: project
---
Experiment 2, iter_1 candidate_3 (2026-11-02): refined `common_divisor_sharpened_exemplar`. The old rule penalized t%g!=0 in multi-example sets. The new rule penalizes a target only when gcd(t, g)<3, where g is the set's GCD (for a single example, g is the example itself). It has no extra parameters.
MAP log-lik on the in-sample data was -2707.9, against -2714.4 for the incumbent. The new rule applied only to singles gave -2712.6, and only to multi-example sets gave -2709.6. Using a threshold of 2 instead of 3 gave -2707.2, but that overlaps with parity. Exempting digit-matching targets gave no help. The smoke fit was -2711.9.
The new rule changes the feature on 306 trials across 10 sets: 90,45; 8,64; 7; 80,70,90,100 and others. Most single-example sets are tested only with factor-sharing targets, so the single_example critique stat has little leverage.
**Why:** this addresses the critique stat single_example_shared_factor_contrast and the 90,45 anomaly.
**How to apply:** see [[common-divisor-sharpened-exemplar]] and [[exp2-iter1-critique-stats]]. background_set_rate_sd remains unaddressed.

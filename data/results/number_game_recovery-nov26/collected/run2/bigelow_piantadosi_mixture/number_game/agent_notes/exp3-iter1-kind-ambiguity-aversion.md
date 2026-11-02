---
name: exp3-iter1-kind-ambiguity-aversion
description: Exp3 iter-1 exploratory slot — rule-vs-range ambiguity aversion on the Weber-heading incumbent, smoke LL -3194.5, submitted
metadata:
  type: project
---
Exp3 iter-1 candidate 0 (2026-10-31): `kind_ambiguity_aversion`. It is the incumbent `heading_extrapolation_weber_width` with p_concept multiplied by (1 - kappa·4·p_rule·(1-p_rule)), where p_rule is the per-set posterior that the concept is a rule rather than a range, and kappa ~ Beta(1.5, 3). It adds no extra compute. The smoke fit gave LL -3194.5 and took 2-4 s, and I submitted it untuned. The idea came from the set 70,10,88,40, which gets 0.13 yes even though every example is even. Considered and dropped: whole-number-line uncertainty, because it needs per-target moments for all 100 targets (too slow), and posterior entropy, which is dominated by the many near-equivalent ranges. See [[exp3-iter1-critique]], [[exp3-iter0-critique]].

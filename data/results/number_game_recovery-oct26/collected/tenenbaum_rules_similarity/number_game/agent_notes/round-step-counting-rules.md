---
name: round-step-counting-rules
description: exp3 iter_4 c0: multiples of 15/20/25/50 as a 6th rule kind on sequence_opens_bottom_ends_top; MAP LL -1.7 vs incumbent, MAP RMSE 0.0058; check OK smoke -3094.3, 240s sampling
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_0 (exploratory, concept-space brief, critiques all null). Submitted `round_step_counting_rules`: the incumbent plus rules for multiples of 15, 20, 25 and 50 as their own kind (N_KINDS=6), made with a string-replace copy.
MAP LL is -3091.4 (incumbent -3089.7). The p_yes RMSE against the incumbent is 0.0058, driven by 90,45|27 and 100|10.
Probes run before choosing: per-set residuals of the incumbent at MAP are noise. Offset progressions (step = gcd of gaps) show no effect in the raw exp3 data (e.g. 15,11: 71 and 91 get 0/5). There are no repdigit sets with more than one example.
**Why:** Concept additions keep coming out null ([[concept-space-saturated]], [[digit-sum-concepts]]).
**How to apply:** Check this model's outcome before adding more rule kinds. Related: [[write-candidate-first]]

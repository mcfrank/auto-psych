---
name: fitted-stray-rate-on-veto-exp3
description: exp3 iter3 cand4 repair — free stray rate on round_landmark_veto +3.8 MAP (0.069 vs fixed 0.1); 7 other one-param screens ≤2.5
metadata:
  type: project
---
2026-10-31, exp3 iter_3/candidate_4_repair_1 (`trusted_example_round_landmark_veto`). The first attempt (veto + spacing) was rejected as a duplicate of round_veto_spacing_sparsity.
- On the incumbent (MAP -3415.3), a free stray rate with prior Beta(2,18) gives -3411.5, with the rate at 0.069. This was submitted. Check OK, smoke -3419.7, 140 s.
- Gains of the other screens:
  - interval Erlang shape exponent +2.5
  - hard 7/8 tables +2.0
  - target-1 endpoint veto +0.6
  - singleton guessing +0.2
  - log nearest-distance logit shift +0.1
  - repeat aversion for targets that are examples 0
- Gotcha: the incumbent's rule rows are 16=mult 3 … 20=mult 7, 21=mult 8.
**Why:** stray rate had been null in exp1, but it revives on the current base.
**How to apply:** scripts are in iter_3/candidate_4_repair_1/scratch (var.py A/B/D/R flags, var2.py E/H/J flags). Related: [[veto-plus-spacing-exp3]], [[spacing-plus-round-veto-exp3]].

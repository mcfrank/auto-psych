---
name: trivial-small-number-avoidance
description: Exp3 r4 repair — penalising targets 1-3 gains ~6.5 MAP nats on leading_digit_string_similarity; set-size/odd-set shifts small
metadata:
  type: project
---
Exp3 r4 (2026-10-18), candidate_4_repair_1. The first attempt (even rule) was rejected as a near-duplicate of leading_digit_even_rule_similarity (RMSE 0.0004), because another slot submitted the same graft that round.
MAP screen on the incumbent (ll −4630.2), single logit terms:
- exp(−(t−1)/2) small-number penalty: −4623.7 (+6.5). t≤3 indicator +6.8; t∈{1,2} +6.2; 1/t +5.6; t=1 only +3.5.
- log k set-size criterion +2.5. All-odd-set shift +1.5.
Submitted as trivial_small_number_avoidance (τ fixed at 2, HalfNormal penalty).
Residuals left over: multiples-of-3 sets are over-predicted (3,81; 3,33,99; 66,93,51), and 4-example sets are under-predicted by 0.022.
**Why:** check novelty against sibling slots too. Grafts of popular cues collide.
**How to apply:** the scripts are in iter_4/candidate_4_repair_1/scratch (mk.py feat dict). See [[exp3-r4-incumbent-screen]].

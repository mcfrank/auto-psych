---
name: hidden-div3-cost
description: exp2 iter_3 cand_5 — shared cost on multiples of 3/6/9/12 on prime_salient_marked_odd, MAP +5.6; other one-param tweaks ~0
metadata:
  type: project
---
2026-10-30, exp2 iter_3/candidate_5 (`hidden_divisibility_by_three`). These are MAP NLL results on prime_salient_marked_odd (base 2123.7):
- Shared cost on the digit-sum rules (multiples of 3, 6, 9, 12; rows 16, 19, 22, 24): 2118.1, cost≈0.69. RMSE vs base 0.023.
- Multiples-of-3 cost alone: 2119.9 (RMSE 0.018).
- Null: all multiples except 5 (2122.8), even cost, fitted stray noise. Fixing the LOG_FACTOR index made it worse (2151.2).
- On this base the lapse fits to ~0 and beta≈0.8.
The scratch screen.py does a fast Powell MAP and prints the top residual items.

**Why:** most of the remaining misfit is scattered item noise (5 responses per item). One-parameter prior tweaks now gain only a few nats.
**How to apply:** expect small gains from here on. Related: [[odd-mult3-set-pattern]], [[incumbent-log-factor-misindexed]].

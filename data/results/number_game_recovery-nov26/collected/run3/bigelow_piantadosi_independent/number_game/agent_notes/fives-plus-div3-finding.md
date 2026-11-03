---
name: fives-plus-div3-finding
description: exp2 iter_4 cand_5 — fives bonus added to hidden_divisibility_by_three gives +16.4 MAP; a shared +r/-r readability parameter gives only +9
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_5 (`fives_readable_digit_sum_divisibility`). MAP NLL on hidden_divisibility_by_three (base 2118.2):
- separate multiples-of-5 bonus (row 18): 2101.8. Fitted bonus≈2.5, div3 cost drops to ≈0.5.
- one shared readability parameter (+r for mult5, -r for div3): 2109.1.
The check printed OK with smoke loglik -2108.7. The smoke fit took ~120 s per 200 NUTS iterations, so admission time is a risk: the earlier fives model on the incumbent timed out ([[fives-bonus-finding]]).

**Why:** the critique's top FDR discrepancy was under-generalised multiples of 5, and the fives bonus closes most of it.
**How to apply:** if this candidate times out, the cause is sampler cost, not lack of signal. Related: [[hidden-div3-cost]].

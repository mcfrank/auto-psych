---
name: round-number-atypical-members
description: exp3 iter2 cand4 — round targets (mult of 10) as atypical concept members +6.6 MAP on set-size incumbent; decade-row ranges null
metadata:
  type: project
---
2026-10-31, exp3 iter_2/candidate_4 (`round_number_atypical_members`). This addresses the critique stat round_target_bonus_no_round_examples (observed -0.057). On the incumbent set_size_evidence_privileged_fives, the MAP loglik is -3423.2 (/tmp/c4i2/mapfit.py, about 25 s).
- One Beta factor rho scales the membership of round targets in every rule except ends-in-0 and contains-0 (rows 25 and 35), and in intervals. It fits rho≈0.85 for +6.6. Applying it to rules only gives +5.1.
- Hundred-chart rows taken as decades (20-29) instead of 21-30: -0.2, null.
- The round over-prediction is driven by even sets, above all 8,64 (observed 0.3 vs predicted 0.65 on all even targets), 12,84,8 and 32,4,16.
**Why:** the residual is a round-target typicality effect, not a range-boundary effect.
**How to apply:** rule row indices are 16-24 multiples, 25-34 ends-in-d and 35-44 contains-d. 8,64 is still the biggest single miss. Related: [[exp3-iter2-critique-stats]].
- Check OK, smoke loglik -3424.6.

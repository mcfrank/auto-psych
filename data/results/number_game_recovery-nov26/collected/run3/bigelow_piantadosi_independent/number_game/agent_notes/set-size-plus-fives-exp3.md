---
name: set-size-plus-fives-exp3
description: exp3 iter1 cand5 — fives bonus grafted on set_size_graded_evidence_weight, smoke loglik -3447.2
metadata:
  type: project
---
2026-10-31, exp3 iter_1/candidate_5 (`set_size_evidence_privileged_fives`). Refinement slot. I grafted the fives bonus (row 18) onto set_size_graded_evidence_weight, which keeps the coupled prime_over_odd. Check printed OK: smoke loglik -3447.2, 93 s per chain. That compares with about -3464 for set-size alone. I did not run a MAP screen.
**Why:** set-size (+9 to +12) and fives (+18) each replicated, and neither live model had both.
**How to apply:** if incumbent slots also add set-size to the fives incumbent, this model is probably dominated. Related: [[fives-on-decoupled-exp3]], [[set-size-evidence-exp3]].

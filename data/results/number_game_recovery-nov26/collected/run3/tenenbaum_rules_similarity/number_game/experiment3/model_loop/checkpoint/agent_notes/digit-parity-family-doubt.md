---
name: digit-parity-family-doubt
description: exp3 iter_3 c4 repair: digit-parity family grafted onto rule_family_doubt_criterion; pool RMSE 0.0042 vs incumbent, ll −1.3 vs incumbent
metadata:
  type: project
---
exp3 iter_3/candidate_4_repair_1 (2026-10-31), digit_parity_family_doubt: the incumbent rule_family_doubt_criterion plus digit_parity_concepts' only-odd/only-even-digit family as a 9th rule family. That family also enters the family-doubt entropy, normalised by log 9. Smoke check OK (36 s sampling). In 2×300 fits: posterior-mean ll −3649.3 vs incumbent −3648.1 and digit_parity −3650.6. Pool RMSE: 0.0042 vs incumbent, 0.0056 vs digit_parity. Shift 0.19.
The first attempt, a size^-s graft ([[specificity-tilted-family-doubt]]), was a duplicate of the incumbent (RMSE 0.00105).
**How to apply:** concept-family additions with a Dirichlet(2) prior reliably pass novelty, even when the data reject them. Script: experiment3/model_loop/iter_3/candidate_4_repair_1/scratch/n/nov.py.

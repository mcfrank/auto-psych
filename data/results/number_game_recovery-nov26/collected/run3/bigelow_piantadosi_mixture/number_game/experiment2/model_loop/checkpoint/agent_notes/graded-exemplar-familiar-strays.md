---
name: graded-exemplar-familiar-strays
description: exp2 iter_3 c3: familiar strays grafted on graded_exemplar incumbent via 16-subset range tables (exact, mix=0 recovers incumbent)
metadata:
  type: project
---
exp2 iter_3/candidate_3 (2026-11-01) is graded_exemplar_familiar_strays. It takes the incumbent graded_exemplar_concept_retrieval and changes the stray distribution to q = (1-mix)/100 + mix*STRAY_RULE_GEN, which adds one param.
Trick: per-example 1/(100 q_i) breaks the e_j tabulation for ranges. Instead, tabulate the sum over ranges of prod_{i in S} w_i for all 16 subsets S, per set and per cell, on the logit(r) grid. The graph then weights each subset by c_|S| * prod rho_i. Rules just scale w by rho before _elementary.
Check passed: smoke ll -2033.8 against the incumbent smoke -2034.9; 200 NUTS iters in 20 s. The range-table prep in prepare_observed dominates the check runtime of about 3.5 min. Novelty was not checked. See [[graded-exemplar-uncertain-reliability]], [[familiar-strays-family-level]].

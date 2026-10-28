---
name: breadth-adaptive-criterion
description: exp3 iter_0 c0: yes-criterion shifts with posterior expected concept breadth; no-conj base; smoke LL -3096.8, ~20s/run
metadata:
  type: project
---
Exp3 iter_0 candidate_0 (2026-10-22), a decision-rule slot. Model: breadth_adaptive_yes_criterion. It is the no-conjunction prime-fixed base, with logit p = logit(belief) + bias - k*(logit(expected breadth) + 1.5), where expected breadth is the posterior · hypothesis coverage. The check returned OK with smoke LL -3096.8 and about 20 s per sampling run.
Exp3 also has 3 stimulus designs: sessions 0-4, 5-9 and 10-14 have identical yes rates within each group. The novelty pool has no trial_index, so per-session models are unsafe.
**How to apply:** Check admission/novelty outcome before reusing. Related: [[decision-rule-mechanisms-null]], [[session-heterogeneity]]

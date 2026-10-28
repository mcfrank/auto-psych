---
name: concept-confusion-decision-noise
description: exp3 iter_3 c2 decision-rule slot: response gain on belief log-odds scaled by posterior entropy over concepts; no-conj base; smoke -3095.6, ~19s/run
metadata:
  type: project
---
Exp3 iter_3 candidate_2 (2026-10-22): concept_confusion_decision_noise. It is built from breadth_adaptive_yes_criterion.py (no-conj, prime-fixed base). It uses logit p = exp(log_gain - slope*(H_norm - 0.3)) * logit(belief), where H_norm is the entropy over 42 rules plus the interval width/decade groups. The check returned OK.
**How to apply:** Check the novelty/ELPD outcome before reusing. Related: [[decision-rule-mechanisms-null]], [[breadth-adaptive-criterion]]

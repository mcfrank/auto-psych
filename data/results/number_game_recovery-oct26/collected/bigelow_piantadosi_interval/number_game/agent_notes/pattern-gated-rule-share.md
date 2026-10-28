---
name: pattern-gated-rule-share
description: iter_4 cand_3 — rule share rising with log set size (+1.8 nats MAP on uneven_rule_familiarity_ends); singleton-only shift null
metadata:
  type: project
---
2026-10-21, experiment3 iter_4 candidate_3. Critique: singleton_rule_far_generalization (the model over-extends rules from one example; q=0.30).
MAP screens on uneven_rule_familiarity_ends (MAP log-lik -3186.3):
- Rule-vs-range log-odds + n_slope*(log n - 0.9): -3184.5 (+1.8), n_slope 0.56, p_yes RMSE 0.0072. Submitted as pattern_gated_rule_share.
- Singleton-only shift: +0.3, a null.
**Why:** the set-size dependence of the rule share is a live lever; a singleton-only bump is not.
**How to apply:** scripts are in iter_4/candidate_3/scratch (va.py, vb.py); mapfit.py is in iter_3/candidate_4/scratch. See [[map-screening-refinements]], [[incumbent-saturation-screens]].

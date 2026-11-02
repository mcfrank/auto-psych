---
name: soft-criterion-decision-rule
description: Logit criterion+gain decision rule on equal-prior digit-concept posterior gives smoke loglik -906.1, same as lapse model
metadata:
  type: project
---
Round 2 candidate_2 (2026-10-30), `soft_confidence_criterion_decision`: p_yes = sigmoid(gain * (logit posterior - logit criterion)), 3 params (lambda_0, criterion, gain). Smoke loglik -906.1 vs -905.8 for `equal_prior_concepts_with_guess_lapse`; fits in ~1 s.

**Why:** response-level mechanisms (lapse, threshold) all land around -906; confirms [[number-game-ceiling]] and [[lapse-small-gain]].
**How to apply:** decision-rule variants are saturated; don't propose more.

---
name: membership-verification-errors
description: exp2 iter_3 c1: graded_exemplar + per-family target-check error eps_f; smoke ll -2037 (no gain over -2034.9), fast
metadata:
  type: project
---
exp2 iter_3 candidate_1 (membership_verification_errors, 2026-11-01). Built on graded_exemplar_concept_retrieval. Rule h endorses target t with eps_f + (1-2 eps_f) r^dmin, where eps_f = 0.5*sigmoid(N(-3.5,1)) per rule family. Ranges have no eps.
Smoke result: ll -2037.0, against -2034.9 for the base model, so no fit gain. It is likely to collapse toward eps→0 or be a near-duplicate, in line with [[exp2-incumbent-refinements]] (lapse-like tweaks collapse). Speed is fine: about 11 s per 200 iters.
**How to apply:** don't retry decision-stage error variants on this incumbent. See [[noisy-criterion-decision-rule]] and [[graded-exemplar-uncertain-reliability]].

---
name: previous-set-concept-intrusion
description: exp3 iter_1 c2: previous trial's set posterior mixed into membership (proactive interference), intrusion≈0.01±0.006, null
metadata:
  type: project
---
exp3 iter_1/candidate_2 (2026-10-31), previous_set_concept_intrusion: kind_uncertainty_shifted_criterion + p_in = (1-w)·member(current set) + w·member(previous trial's set, same target); w=0 on trial_index 0. Sets are interleaved in exp3 (only 1265/6750 rows repeat the previous set), so this is identifiable.
Quick 1×200 fit: intrusion 0.009±0.006, shift 0.25±0.20; smoke ll −3653.6 (incumbent ≈ −3653.3). Smoke check OK in ~100 s.
Fit script: /tmp/c2fit.py <candidate.py> <responses.csv> prints param means.
**Why:** cross-trial carry-over is another null in exp3 (cf. lag1_response_repeat null in [[experiment3-iter0-critique-stats]]); novelty vs incumbent may fail.
**How to apply:** don't re-propose trial-to-trial interference variants without a new angle.

---
name: parity-representativeness
description: exp3 iter_1 c2: parity-mixed concepts penalised per repeated shared parity (representativeness); smoke -3094.1, 202 s
metadata:
  type: project
---
2026-10-22, exp3 iter_1 candidate_2 `parity_representativeness_bias`. It targets the critique stat parity_match_outside_span_gap (q 0.096). Built on a copy of last_example_marks_range_top. It adds a HalfNormal penalty times (n-1) for all-same-parity sets, applied to concepts that contain both parities (RULE_MIXED rules, intervals of width >=2, conjunctions by their slot rule). Check OK: smoke LL -3094.1, 202 s per sampling run.
Pitfall: indexing a numpy array with a pm.Data tensor fails; wrap it first with pt.as_tensor_variable.
**How to apply:** Check novelty and ELPD outcome before reusing. Related: [[exp3-iter1-critique-stats]]

---
name: familiar-strays-anchor
description: exp2 iter_2 c4: incumbent familiar_strays_family_level_prior + first-example anchor a1; elementary-polynomial layout with first_onehot, verified vs brute force
metadata:
  type: project
---
exp2 iter_2/candidate_4 (2026-11-01) is familiar_strays_first_example_anchor. It grafts a1 from first_example_anchored_reliability onto the familiar-strays incumbent.
Implementation: keep sorted examples (for interval runs) plus a first_onehot. e_j runs over the later examples only, and c_table is indexed by m = n - 1. The factor on each subset is (1 - a1) + a1 * [first in subset] * inv_q1 / s. For intervals, run_pow needs j = 0..5, because the anchor adds one extra s^-1. Brute-force check err 1e-13. Check OK: smoke 32 s/200 iters, loglik -2035.6.
Gotcha: model.compile_fn(..., point_fn=True, on_unused_input="ignore") mis-maps inputs and returns garbage. Use a non-point fn over a list of outputs instead.
**How to apply:** check the ELPD/novelty outcome of this slot before retrying an anchor graft. See [[first-example-anchored-reliability]], [[familiar-strays-family-level]].

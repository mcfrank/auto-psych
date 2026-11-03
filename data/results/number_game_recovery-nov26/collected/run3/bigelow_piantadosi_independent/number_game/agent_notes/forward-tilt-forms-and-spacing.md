---
name: forward-tilt-forms-and-spacing
description: exp3 iter4 cand3 — forward tilt shape variants null; spacing shift on forward_continuation_response_bias +4.2 MAP, smoke -3416.3
metadata:
  type: project
---
2026-10-31, exp3 iter_4/candidate_3 (`forward_continuation_spacing_sparsity`). Base forward_continuation_response_bias MAP loglik -3411.6.
- Variants of the tilt's form are all null or worse: decays with distance -0.5, grows with distance -0.9, span-scaled decay -0.6, last-shown-example reference +0.4, tilt plus last-example term +1.1.
- Grafting the mean-gap spacing logit shift gives -3407.4 (+4.2). Submitted. Check OK, smoke -3416.3, 120 s sampling.
- The iter4 critique had 0 significant stats.
**Why:** the upward lean is a flat shift and its form cannot be improved. Spacing still adds on top of it.
**How to apply:** the screen script is /tmp/c3i4/var.py (VAR env). The novelty risk is overlap with round_veto_spacing_sparsity, which differs only by the tilt. Related: [[forward-continuation-response-bias]], [[spacing-plus-round-veto-exp3]].

---
name: widest-gap-sparsity-exp3
description: exp3 iter4 cand5 — widest-gap sparsity on round_veto_spacing_sparsity +1.7 MAP over mean gap; other spacing forms and singleton/isex tweaks null
metadata:
  type: project
---
2026-10-31, exp3 iter_4/candidate_5 (`widest_gap_sparsity_round_veto`). MAP loglik on round_veto_spacing_sparsity (mean gap) was -3410.1. Alternatives:
- max gap: -3408.3 (submitted). RMS gap: -3408.9.
- Forward tilt graft: -3407.5, not submitted because the incumbent-refinement slots probably do that.
- Null or worse: log gap, min gap, log-ratio gap, span, span/n, gaps including the 1/100 edges, spacing only outside or only inside the span, examples-as-target exemption, singleton term.
- Lapse fits to 0.
**Why:** the spacing family is almost saturated, and gains here are about 2 nats.
**How to apply:** scripts are in /tmp/c5i4 (var.py with the V env flag, map.py). Related: [[spacing-plus-round-veto-exp3]], [[forward-continuation-response-bias]].

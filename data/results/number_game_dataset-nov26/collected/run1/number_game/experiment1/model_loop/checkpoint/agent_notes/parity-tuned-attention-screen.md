---
name: parity-tuned-attention-screen
description: Round 2 MLE screen — adding parity to variability_tuned_exemplar_attention gains ~13 nats in-sample
metadata:
  type: project
---
Round 2, cell_1 (2026-11-01): in-sample NLL, incumbent variability_tuned_exemplar_attention 1574.6; + fixed parity penalty 1561.3; + parity penalty scaled by parity homogeneity of the set 1561.9 (same param count); + exp(a+b*homog) parity 1561.0 (degenerate ridge, b→136). Submitted the homogeneity-scaled version as candidate_3 `parity_tuned_exemplar_attention`; check_candidate OK.

**Why:** critique flagged parity_contrast_far_nodigit (q=0.016). **How to apply:** don't re-screen parity on this incumbent; next ideas: lapse on top. Related: [[exemplar-variant-mle-screen]], [[adaptive-attention-exemplar-screen]].

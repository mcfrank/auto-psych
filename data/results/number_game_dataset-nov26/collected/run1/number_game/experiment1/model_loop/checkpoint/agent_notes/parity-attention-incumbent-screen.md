---
name: parity-attention-incumbent-screen
description: Round 2 MLE screen — parity dim on variability_tuned_exemplar_attention gives ~13 nats; homogeneity-scaled parity is best-identified
metadata:
  type: project
---
Round 2 cell_1 (2026-11-01), in-sample NLL on incumbent variability_tuned_exemplar_attention: base 1574.6; + fixed parity penalty 1560.9 (degenerate ridge: huge scales, beta→0); + parity penalty only when all examples share parity 1561.0; + penalty ∝ parity homogeneity 1561.9 (sensible params). Submitted the last one as iter_2 candidate_4 `variability_tuned_parity_attention`; check OK. Motivated by the critique stat parity_contrast_far_nodigit (q=0.016).

**Why:** saves re-screening. **How to apply:** builds on [[exemplar-variant-mle-screen]] and [[adaptive-attention-exemplar-screen]]; a lapse term is still untested on this base.

---
name: shared-factor-exemplar-screen
description: iter_3 cand_5 MLE screen on variability_tuned_parity_attention — coprime penalty -7.4 nats; set-size criterion/sharpening no gain
metadata:
  type: project
---
iter_3 candidate_5 (2026-11-02), in-sample NLL on variability_tuned_parity_attention (base 1561.9; fit is degenerate: beta≈0.06, large attention scales, predictions on flat levels):
- + fixed penalty when target and example are coprime: 1554.5 → submitted as `shared_factor_parity_exemplar` (check OK, smoke LL -1558.5)
- + penalty when they share none of {3,5,7}: 1559.6; + penalty when target lacks the set's gcd: 1558.2
- criterion shifted by log(set size): 1561.9; magnitude sharpening by set size: 1561.8 (no gain)
- set-cohesion criterion (target sim relative to within-set sim): untested (slow; timed out)
None of these moves the across-set yes-rate SD (0.147 predicted vs 0.193 observed). Biggest set residuals: digit-homogeneous sets and powers of 2 are over-predicted, while 3,63 (→36,39), 6,66, 100 (→90) and 55,15,75,45 are under-predicted, which points to arithmetic rules.

**Why:** saves re-screening. **How to apply:** see [[iter3-critique-stats]], [[parity-attention-incumbent-screen]].

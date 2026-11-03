---
name: exp2-iter0-separate-specificity
description: Exp2 iter-0 cand_5 repair — range-only occam was dup of narrow_range (RMSE 0.0004); added separate rule breadth exponent h, MAP h≈-0.06 (+0.5 nats), RMSE ~0.006 vs h=0
metadata:
  type: project
---
Exp2 iter 0, candidate_5_repair_1 (2026-10-30). The first attempt (`occam_range_width_prior`) was rejected as a near-duplicate of `narrow_range_specificity_bias` (RMSE 0.00044).
For the repair I submitted `separate_rule_and_range_specificity`: the range width prior plus a separate rule prior ∝ size^-h. It keeps all 45 rules separately (the tensor is cells × 45).
- Sensitivity at MAP: h is sharply identified. The optimum is about -0.06 (slightly favours broad rules) and gains +0.5 nats. h=±0.1 changes p_yes by an RMSE of 0.011.
- Check OK: smoke loglik -1930.1, about 2.5 min.

Pitfall: a pytensor compile_fn output buffer gets reused between calls. Copy the outputs, or build the function with pytensor.function over value_vars and pass inputs positionally.

**How to apply:** to escape the novelty gate, add a parameter with a nonzero sharp optimum, then check the RMSE vs the parent at a perturbation of about one posterior sd. See [[exp2-iter0-range-occam]].

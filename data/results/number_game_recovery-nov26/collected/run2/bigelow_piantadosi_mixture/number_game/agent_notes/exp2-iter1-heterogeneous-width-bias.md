---
name: exp2-iter1-heterogeneous-width-bias
description: Exp2 iter-1 cand_3 — MAP screens on narrow_range incumbent (NLL 1927.3); population spread of gamma +0.8 submitted; endpoint-position range priors +1.8 but near rel-width
metadata:
  type: project
---
Exp2 iter 1, candidate_3 (2026-10-30). I refined `narrow_range_specificity_bias`. My MAP refit of it gives NLL 1927.33 (lambda logit -0.77, gamma -0.12, guess→1). Screens (scratch/screen*.py):
- logit-lambda population spread (Gauss-Hermite): 1926.87, RMSE 0.0057 vs the incumbent.
- gamma population spread: 1926.50 (tau≈0.5), RMSE 0.008. **Submitted** as `heterogeneous_range_specificity_bias`, with 7 GH nodes vectorised with batched_dot (grad 12 ms, same as incumbent; a python loop over nodes was ~6x slower).
- range prior with a linear term in the lower endpoint: 1925.52 (gamma→0). A linear term in the midpoint: 1925.71. Both are only RMSE 0.004-0.005 from the relative-width model, so they risk being duplicates.

**How to apply:** individual-difference (quadrature) spreads are a cheap way to answer "excess disagreement" critiques, but they gain under 1 nat. See [[exp2-iter1-critique]], [[number-game-ceiling]].

---
name: exp3-iter3-set-size-evidence-weight
description: Exp3 iter-3 cand_3 on the WR incumbent — set-size stray rate and position trust tie; log-lik scaled by n^-kappa gives +1.34 MAP (kappa≈-0.03), submitted via a 4-D Taylor axis
metadata:
  type: project
---
Exp3 iter 3, candidate_3 (2026-10-31), refinement slot on heading_weber_upward_with_replacement. Base MAP LL is -3190.30.

The critique's only hit, prefix_extension, comes from one pair of sets (31,3,1 vs 31,3,1,15) with 5 answers per cell, so it is noise-level.

MAP screens (ΔLL):
- stray rate shifted by set size: +0.2
- position-graded trust: +0.1
- log-lik × n^-kappa: +1.34 (kappa -0.032, RMSE 0.008)

Submitted `set_size_scaled_evidence_weight`. Smoke -3192.7. 200 iterations take 25 s; the gradient takes 12.5 ms versus 8 ms for the incumbent.

**How to apply:** tempering breaks the precomputed exp(LL) moments. Add a Taylor axis in (t-1)·L' with L'=L-Lmax, using K_T=5 and dropping intervals below L'<-20. Use pt.dot(moments, coef.T)[rows, cls] rather than elementwise gathers, which are 3x slower. A dense softmax over 5113 hypotheses costs about 84 ms per gradient. Scratch is in iter_3/candidate_3/scratch (scr.py does the screens). See [[exp3-iter2-with-replacement-swap]], [[fit-speed-budget]].

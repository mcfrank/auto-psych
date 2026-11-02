---
name: exp3-iter3-set-size-evidence
description: Exp3 iter-3 cand_4 — set-size-scaled evidence (n/2.5)^-rho on the WR incumbent +1.35 MAP, RMSE 0.0088; stray-by-size/position null; 4-D Taylor (evidence axis) speed trick
metadata:
  type: project
---
Exp3 iter 3, cand_4 (2026-10-31), refinement slot on heading_weber_upward_with_replacement. The base MAP LL is -3190.30.

MAP screens:
- stray rate by set size: +0.2
- position-graded stray rate: +0.1
- last-example trust: +0.06
- evidence scaled by (n/2.5)^-rho: +1.35 (rho=-0.068, so larger sets weigh more per example; this runs opposite to the prefix-extension critique). Pool RMSE 0.0088.

Submitted `set_size_scaled_evidence`. Smoke -3192.4.

Speed: a dense softmax over 5113 hypotheses is about 85 ms per gradient, too slow. Instead, add a 4th Taylor axis in (w-1)*(ll-scale)/10 to the incumbent's moments. With K_G=7, K_R=6, K_E=7 it takes 13.7 ms and is exact to 1e-8 at MAP. With K=12/10/10 the GEMMs were memory-bound at 76 ms.

The screening script is at iter_3/candidate_4/scratch/s/screen.py. See [[exp3-iter2-rule-breadth-graft]], [[fit-speed-budget]].

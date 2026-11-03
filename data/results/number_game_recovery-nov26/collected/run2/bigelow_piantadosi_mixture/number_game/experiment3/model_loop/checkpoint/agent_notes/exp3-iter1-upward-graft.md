---
name: exp3-iter1-upward-graft
description: Exp3 iter-1 cand_3 — Weber-heading incumbent MAP 3189.9; upward/below/open/overall-direction/range-size-exponent screens all ≤0.7 nats; upward-room graft submitted (3-D Taylor)
metadata:
  type: project
---
Exp3 iter 1, cand_3 (2026-10-31), refinement of `heading_extrapolation_weber_width`. Numpy MAP NLL 3189.9 (gamma -0.08, beta 0.04, lapse→0, guess_yes→1).
Screens (Δ nats): upward room +0.6 (delta 0.23), below room +0.58, overall first→last direction +0.67, open-ended +0.16, range size-principle exponent rho→0.99 (null). Per-set residuals look like noise (max |z| 3.2 on 73,33,3).
Submitted `heading_weber_with_upward_counting`: exponent (delta + h beta)*above - h beta*below, i.e. a 3-D Taylor (12×10×10 moments; 4.2 ms/grad vs incumbent 1.1) with one coefficient vector per heading class. MAP p_yes RMSE vs incumbent 0.0056; matches numpy screen to 1e-15.
Speed pitfall: gathering coef[cell_set] (cells×moments) was 46 ms/grad; use pt.dot(member_moments, coef_classes) then pick column → 9 ms. Direct dense 100×100 per-set cumsum is 140 ms/grad, so it is too slow; use moments.
Scratch: experiment3/model_loop/iter_1/candidate_3/scratch/screen.py (generic extra-feature MAP harness). See [[exp3-iter0-weber-heading]], [[exp3-iter0-upward-room]].

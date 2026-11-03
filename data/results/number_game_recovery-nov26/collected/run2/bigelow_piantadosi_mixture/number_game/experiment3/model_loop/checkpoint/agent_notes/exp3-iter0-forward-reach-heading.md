---
name: exp3-iter0-forward-reach-heading
description: Exp3 iter-0 cand_4 — on exp3 the heading strength is about 0 and every refinement screen ties within 0.7 nats; forward-reach heading submitted
metadata:
  type: project
---
Exp3 iter 0, candidate_4 (2026-10-31), a refinement slot on heading_direction_range_extrapolation. The incumbent's numpy MAP NLL is 3190.9, with **beta≈0.04**: the heading cue is nearly inert on exp3. guess_yes→1 at a lapse of 0.007.
Screens (MAP, all within 0.7 nats):
- heading forms: only-if-last-extreme, net trend (beta -0.18), mean step sign, room vs last example, log room.
- room ahead only: +0.64, beta 0.32.
- free stray rate alpha: sharply peaked at 0.9 (0.85 loses 26 nats, 0.95 loses 60).
- rule-vs-range odds temperature: tau 0.95, +0.5.
- step-over-span: worse.
Calibration is perfect across p bins and prule bins. Most cells have n=5, so the residuals are noise.
Submitted `forward_reach_heading_ranges`: c = room ahead/100 in the heading direction, with no penalty for room behind. N_TERMS 16. Novelty RMSE 0.005. Smoke -3193.9, 40 s.
Pitfalls: `pkill -f screen.py` kills the calling shell too. Nelder-Mead/L-BFGS-B stall here; use Powell from the incumbent MAP.
See [[exp2-iter2-heading-direction]], [[exp3-iter0-critique]].

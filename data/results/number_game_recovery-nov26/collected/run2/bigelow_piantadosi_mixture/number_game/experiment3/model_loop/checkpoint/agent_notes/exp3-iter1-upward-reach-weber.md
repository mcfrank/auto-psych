---
name: exp3-iter1-upward-reach-weber
description: Exp3 iter-1 cand_4 — digit-product gain is one set (34,26); upward room replaces heading on Weber incumbent, MAP +0.55, submitted
metadata:
  type: project
---
Exp3 iter 1, candidate_4 (2026-10-31), refinement slot on `heading_extrapolation_weber_width`.
- Digit-product family (iter-0 screen +1.35) is driven by a single set, 34,26 (product 12), so it is spurious, like digit difference. Don't graft it.
- Submitted `upward_reach_weber_width`: the incumbent with its heading room swapped for order-free room above (clip(hi - max, 0)/100, from upward_room_range_prior). Same Taylor machinery and no new params.
- MAP response LL is -3190.30 vs the incumbent's -3190.85. Beta is 0.21 (the incumbent's heading beta is 0.04). p_yes RMSE vs the incumbent is 0.0055. Smoke -3192.8 in 36 s.
- MAP pitfall: `compile_logp()(mp)` fails on find_MAP output (it wants transformed names). Compute the Bernoulli LL from mp['p_yes'] directly.
See [[exp3-iter0-upward-room]], [[exp3-iter0-weber-heading]].

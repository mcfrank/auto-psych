---
name: exp3-iter3-trend-origin-room
description: Exp3 iter-3 cand_5 — prefix stat is one set pair; trust slope / set-size lapse null; sample-size neglect and per-set stray schedules lose 50-110 nats; first→last trend heading swap submitted (smoke -3192.7)
metadata:
  type: project
---
Exp3 iter 3, candidate_5 (2026-10-31), refinement slot. Base: heading_weber_with_upward_counting, MAP obs LL -3190.24.

- The prefix_extension critique stat rests on a single pair: 31,3,1 → 31,3,1,15 (15 targets, yes rate 0.33 → 0.62). Not worth building a mechanism around.
- Position-graded trust, using an exact expansion over subsets of genuine examples: slope 0.047, +0.14 nats, and 65 ms per gradient (7x the base). Null, and too slow.
- Lapse rate as a logistic function of set size: slope -0.05, null.
- Tempering by a fixed set-size schedule (sample-size neglect): min(n,3) -53, n^0.8 -49, 1+log n -150.
- Stray rate is sharp at 0.9. Per-set stray schedule 0.9^(1/n): -113. 1-0.1n/2.5: -38. Constant 0.85: -30. Constant 0.93: -15.
- Submitted: heading = sign(last - first) in place of the last step. MAP -3190.06, beta -0.13, read as "the run started partway into the range". Named `trend_origin_room_upward_counting`, smoke -3192.7, about 15 s.
**How to apply:** the stray and evidence-weight dimensions are saturated on exp3, so don't re-screen them. See [[exp3-iter3-critique]], [[exp3-iter2-rule-breadth-graft]].

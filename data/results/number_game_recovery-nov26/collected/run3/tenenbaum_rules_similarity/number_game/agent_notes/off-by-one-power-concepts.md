---
name: off-by-one-power-concepts
description: exp2 iter_1 cand_2 — near-miss rule family (squares/cubes/powers ±1) added to beta-share incumbent; check passed, smoke ll -2270
metadata:
  type: project
---
exp2 iter_1 candidate_2 (2026-10-30): off_by_one_power_concepts = beta_share_width_exception_concepts + a 9th rule family of 10 "power/square/cube ±1" rules (e.g. 1,3,7,15,31,63). Motivated by exp2 sets 31,3,1(,15), 33,17,5,9, 33,3,65, 7,63, 3,7 — but few targets probe them (31,3,1,15 → 63 only 2/5 yes). Check passed; smoke-fit ll -2270.0. Expect a small gain at most, as in [[concept-space-additions-map]].
**Why:** the brief asked for a concept-space change; exp2's stimulus sets include shifted powers of 2.
**How to apply:** once admission/LOO results are in, record whether the near-miss family earned any weight before trying more concept additions.

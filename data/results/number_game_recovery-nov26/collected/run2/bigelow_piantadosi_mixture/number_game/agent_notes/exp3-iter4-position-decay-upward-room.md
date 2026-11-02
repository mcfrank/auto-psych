---
name: exp3-iter4-position-decay-upward-room
description: Exp3 iter-4 cand_4 — upward-room range prior grafted on position_decaying_example_evidence via 16-term Taylor moments per (set,pattern,width) class; smoke -3192.6, 18 ms/grad, submitted
metadata:
  type: project
---
Exp3 iter 4, candidate_4 (2026-10-31), a refinement slot on the incumbent position_decaying_example_evidence. Submitted `position_decay_upward_room`: the incumbent with its prior over intervals changed to exp(delta*room_above/100)/Z.
- Implementation: the parent's exact class trick is kept. Each class and each (cell, class) entry carries K=16 moments sum(room^j); class weight = moments·(delta^j/j!), and Z comes from per-set moments. Prep takes 4 s; a gradient takes 18 ms.
- Smoke LL -3192.6, against the incumbent's -3192.4. I did not screen MAP or novelty RMSE; the prefix-broadening critique stat (one set pair) was not targeted ([[exp3-iter3-trend-origin-room]]).
See [[exp3-iter3-position-decaying-evidence]], [[exp3-iter0-upward-room]].

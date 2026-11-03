---
name: exp3-iter1-two-sided-room-weber
description: Exp3 iter-1 cand_5 repair — upward_room+Weber was a dup (pool RMSE 0.001); separate above/below room on Weber, no heading, MAP +0.64, submitted
metadata:
  type: project
---
Exp3 iter 1, cand_5 repair (2026-10-31). The first attempt, `upward_room_weber_width`, was rejected as a near-duplicate of heading_weber_with_upward_counting: pool RMSE 0.00102, against an observed-cell RMSE of about 0.002. So pool RMSE ≈ half of observed-cell RMSE. Aim for ≥0.004 on observed cells.

MAP screens on the Weber base with heading pinned to 0 and room above included (Δ vs ref 3189.30):
- open: 0
- lopsidedness: 0
- first-shown-end room: 0
- sqrt(above): +0.1
- probe as weak example: +0.3 (w≈0.01)
- separate room below: +0.64 (delta 0.235, epsilon 0.543), RMSE 0.0063 vs ref and 0.0055 vs abs-width symmetric margin + open

Submitted `two_sided_room_weber_width`, a refinement of heading_weber_with_upward_counting with heading swapped for room below. It uses a 3-D Taylor with one coefficient vector. Smoke -3192.6 in 7 s.

Pitfall: `pkill -f` again killed the shell (exit 144). Kill by PID instead.
Scripts are in iter_1/candidate_5_repair_1/scratch (s1 = feature screens, s2 = probe, s3 = novelty vs margin).
See [[exp3-iter1-upward-graft]], [[exp3-iter0-weber-heading]].

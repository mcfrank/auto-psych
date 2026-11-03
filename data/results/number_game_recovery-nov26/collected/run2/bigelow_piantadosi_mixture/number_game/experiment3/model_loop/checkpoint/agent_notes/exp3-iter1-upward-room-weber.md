---
name: exp3-iter1-upward-room-weber
description: Exp3 iter-1 cand_5 — Weber relative-width prior grafted onto upward_room_range_prior, smoke -3193.0, submitted; novelty RMSE not checked
metadata:
  type: project
---
Exp3 iter 1, cand_5 (2026-10-31), refinement slot. The iter-1 critique found 0 of 8 statistics significant, so there was no discrepancy to target. I grafted the incumbent's (w/mid)^gamma width prior onto `upward_room_range_prior`, keeping log_open and the room-above credit beta.

Implementation: a 2-D Taylor moment table (K_G=28 × K_B=24), split into bounded and open ranges (2×28×24 columns), with coef = [joint, e^log_open·joint]. The smoke run gave -3193.0 in about 7 s per chain. I did not check the novelty RMSE against the parent, and the swap may be too small to pass the gate.

See [[exp3-iter0-upward-room]] and [[exp3-iter0-weber-heading]].

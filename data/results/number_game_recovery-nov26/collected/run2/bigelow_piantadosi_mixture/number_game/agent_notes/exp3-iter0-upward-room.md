---
name: exp3-iter0-upward-room
description: Exp3 iter-0 cand_5 — stray 0.9 sharply optimal again, MAP screens of margin variants all tie (~-3190); upward-only room prior submitted
metadata:
  type: project
---
Exp3 iter 0, cand_5 (2026-10-31), refinement slot. MAP response LL on exp3:
- Model screens: probe_with_margin -3189.8 (w_probe≈0.03, so the probe mechanism is dead here); symmetric margin -3190.1; above-only room -3190.2; relative-margin (room as a fraction of the range) -3191.0; (above-below) -3190.8; below-only -3191.4; heading incumbent -3191.8 (heading beta≈0.04, so it is dead on exp3).
- Stray rate, fixed 0.8/0.95/0.7: -3246/-3226/-3324. So 0.9 is exact, and freeing it would be a dup.
- The session yes-drift (0.49→0.39) is fully explained by the stimuli (the MAP fit matches every quintile). The cell chi2 is 1158 over 1307 cells, i.e. at the ceiling.

I submitted `upward_room_range_prior` (the margin model with c = room above/100). Its RMSE against the symmetric margin is 0.005; smoke -3193.1, 21 s per chain.
p_yes eval pitfall: use `model.replace_rvs_by_values` ([[iter3-free-outlier]]). The screener is iter_0/candidate_5/scratch/resid.py. See [[number-game-ceiling]].

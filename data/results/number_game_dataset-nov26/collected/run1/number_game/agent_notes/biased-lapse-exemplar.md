---
name: biased-lapse-exemplar
description: exp3 iter0 c0 repair — incumbent round_number_averaging + lapse with yes-bias guess; MAP +1 nat, RMSE 0.0056 vs incumbent
metadata:
  type: project
---
Replaced the drifting-criterion candidate ([[drifting-criterion-exemplar]]), which was rejected as a near-duplicate (RMSE 0.00065): drift on trial_index can't be seen on bare stimuli. The new candidate: p = (1-lapse)*sigmoid(...) + lapse*guess_yes, with lapse ~ Beta(2,18) and guess_yes ~ Beta(2,2). MAP: lapse 0.056, guess 0.33, loglik -4107.2 vs incumbent -4108.2. Smoke -4114.3.
**Why:** a decision-level change needs to move predictions on stimuli alone to pass the novelty gate (≥0.002 RMSE). Lapse moves them by about 0.0056 on the training stimuli.
**How to apply:** a lapse adds little fit (about 1 nat), so don't expect decision-rule tweaks on this incumbent to win.

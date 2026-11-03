---
name: session-drift-null-finding
description: experiment2 trial_index drift critique: logit/disengage/commitment drifts on incumbent gain ~0 at MAP; repeat aversion ~+1.5
metadata:
  type: project
---
Tried 2026-10-30 (experiment2 iter_0 candidate_0) on costly_squares_final_digit_concepts, MAP log-post -2231.6:
- Linear logit criterion drift: MAP drift 0.39, +0.06. Power-onset drift + bias: no gain. Disengage-to-no rising over session: worse. Beta (commitment) drifting by 90-trial block: drift -0.02.
- The yes-rate drop is set-confounded: residuals are set-specific, and the last block under-predicts mid-p items (0.25 vs 0.43) without a consistent trend.
- Repeat aversion (examples-as-targets yes x (1-r)): best r ~0.05, about +1.5. The incumbent already gives p=0.95 on the "no" items, so the ceiling is ~9 nats.
- Note: trial_index must default when missing (novelty pool rows lack it), otherwise the model is rejected.
- pm.find_MAP from the default start lands in bad optima; start from the incumbent's MAP.

**Why:** the critique's q=0.048 looks like a weak signal ([[experiment2-data-quirks]]).
**How to apply:** don't spend candidates on session drift unless it has per-participant structure.

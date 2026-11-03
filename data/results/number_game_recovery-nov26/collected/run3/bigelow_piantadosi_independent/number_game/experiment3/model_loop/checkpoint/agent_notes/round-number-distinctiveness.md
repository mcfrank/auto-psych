---
name: round-number-distinctiveness
description: exp3 iter2 cand3 — rejecting unsupported round targets +7.9 MAP loglik on set_size_evidence_privileged_fives; decade-row range variants null
metadata:
  type: project
---
2026-10-31, exp3 iter_2/candidate_3 (`round_number_distinctiveness`). The incumbent's MAP loglik is -3423.2. The change: when no example is round, p_raw is multiplied by (1 - rho) for round targets. It fits rho = 0.18 and gives -3415.3, which matches the critique statistic (-0.057). Applying rho to every round target that is not an example gives only -3416.9. Using decade blocks (20-29) as the hundred-chart rows, or decades together with rows, is null (within ±0.5).
Where the gap was: in all-even sets with no round example, round targets get 0.39 yes observed vs 0.50 predicted, while non-round evens fit (0.45 vs 0.46). Most of it comes from 8,64, where the model predicts about 0.65 for every even number and people give about 0.3.
**Why:** this saves re-screening round and range variants.
**How to apply:** the round-number cue lives at the target stage. Range representation does not explain it. Related: [[exp3-iter2-critique-stats]].

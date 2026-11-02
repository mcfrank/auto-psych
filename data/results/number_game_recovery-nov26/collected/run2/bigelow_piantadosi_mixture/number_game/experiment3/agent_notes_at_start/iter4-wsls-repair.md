---
name: iter4-wsls-repair
description: Round 4 cand_2 repair — description-length cost was a near-duplicate (RMSE 0.0014); win-stay/lose-shift single-concept testing NLL 950.5 vs 902.9 (~48 nats worse), submitted as distinct
metadata:
  type: project
---
Round 4, candidate_2 repair (2026-10-30). `description_length_rule_cost` was rejected because it predicts like the equal-prior without-replacement model (RMSE 0.00137 < 0.002). The cost parameter shrinks toward 0.
The repair is `win_stay_lose_shift_concept_testing`. It holds one concept, keeps it while each example is a member, and resamples from the WOR posterior on a miss. MAP NLL is 950.5, against 902.9 for the incumbent. It is clearly novel but likely to be pruned. The grad takes 2.9 ms and the smoke loglik is -952.7.
**Why:** near-incumbent tweaks fail the novelty gate, and distinct process models lose ([[number-game-ceiling]], [[iter4-process-screens]]).
**How to apply:** pure WSLS (no re-weighing of surviving concepts) is ruled out. Before submitting, check the novelty RMSE against the nearest model at MAP.

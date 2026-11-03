---
name: accumulating-round-skip
description: exp3 iter2 cand3 repair — round veto growing with # non-round examples, 1-(1-r)^n; +6.8 MAP, 1 nat worse than fixed veto
metadata:
  type: project
---
2026-10-31, exp3 iter_2/candidate_3_repair_1 (`accumulating_round_skip_evidence`). The first attempt (a fixed veto on unsupported round targets) was rejected as a duplicate of round_landmark_veto (RMSE 0.00035).
- Repair: p *= (1-r)^n for round targets when no example is round, where n is the set size. MAP loglik -3416.4 (r≈0.061), against the incumbent's -3423.2 and the fixed veto's -3415.3.
- Check OK, smoke loglik -3424.6. Novelty was not verified locally.
**Why:** the round-number critique slot is crowded (veto, unrepresentativeness, decade blocks).
**How to apply:** a set-size-graded round veto does not fit better than a fixed one. Don't bother with more round variants. See [[round-landmark-veto-exp3]].

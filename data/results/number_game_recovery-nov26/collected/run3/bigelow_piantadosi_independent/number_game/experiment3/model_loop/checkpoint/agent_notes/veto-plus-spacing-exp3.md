---
name: veto-plus-spacing-exp3
description: exp3 iter3 cand3 — spacing logit shift on round_landmark_veto +5.3 MAP; single-digit/fives/graded round vetoes null
metadata:
  type: project
---
2026-10-31, exp3 iter_3/candidate_3 (`round_veto_spacing_sparsity`), a refinement of the incumbent round_landmark_veto. The incumbent's MAP loglik is -3415.3, and its lapse fits to about 0.
- Adding the mean-gap logit shift from example_spacing_sparsity_bias gives -3410.1 (coef -0.57, veto 0.18). This was submitted.
- Null variants:
  - a single-digit-target veto after all-two-digit sets (fits to 0.002), with the critique's p=0.07 statistic
  - a shared round/single-digit veto: -1.4
  - multiples of 5 as the landmarks: -3.7
  - an extra veto on targets ending in 5: 0
  - a veto graded by the fraction of non-round examples: -0.4
**Why:** the round-veto variants are exhausted, and the iter3 critique has no significant statistics.
**How to apply:** screens are in /tmp/c3i3 (map.py, base/B-G.py). Related: [[spacing-on-set-size-exp3]], [[round-landmark-veto-exp3]], [[exp3-iter3-critique-stats]].
- Check OK, smoke loglik -3418.6.

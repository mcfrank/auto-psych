---
name: spacing-plus-round-veto-exp3
description: exp3 iter3 cand4 — spacing logit shift grafted on round_landmark_veto +5.3 MAP (additive); single-digit-target veto null
metadata:
  type: project
---
2026-10-31, exp3 iter_3/candidate_4 (`sparse_spacing_round_landmark_veto`). Incumbent round_landmark_veto MAP loglik -3415.3 (screen /tmp/c4i3/map.py, VAR env).
- Spacing mean-gap logit shift (coef -0.57) + round veto (0.18): -3410.1, +5.3. The two effects are additive.
- Single-digit target veto when no example ≤9 (iter3 critique top stat, z -1.8): fits to 0, shared with round veto -1.4. Null.
- Iter3 critique: 0 significant stats. Biggest residual sets: 8,64 (obs 0.25 vs 0.53), 14,98,91 (0.2 vs 0.57), singletons under-predicted.
**Why:** remaining misfit is wide-spaced sets with low yes overall.
**How to apply:** if another slot also grafts spacing onto the veto, this is a duplicate. Related: [[spacing-on-set-size-exp3]], [[round-landmark-veto-exp3]].

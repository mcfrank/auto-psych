---
name: whole-set-round-category
description: exp3 iter0 c2: round kinship only when all examples end in 0/5, on round_number_averaging incumbent, +10 nats MAP, smoke -4104.5
metadata:
  type: project
---
exp3 iter0 candidate_2 (exploratory, exemplar brief): `whole_set_round_category` = incumbent round_number_averaging_exemplar with round_both gated by ALL examples ending in 0/5 (no new params). MAP -4098.3 vs incumbent -4108.2. Targets the mixed_set_round_target_contrast critique ([[exp3-iter0-critique-stats]]).
Screened and dropped (MAP vs incumbent): similarity-cued single-example retrieval, i.e. probability averaging over examples (-36 nats; uniform -43); all-odd parity shift +2.4; magnitude kernel narrowing with n ~0; round fraction^2 gate +8.5; round term added once per set instead of per example ~0.
**Why:** records which pooling/round variants have been ruled out.
**How to apply:** don't re-screen probability-averaging retrieval; whole-set round gating is the useful direction.

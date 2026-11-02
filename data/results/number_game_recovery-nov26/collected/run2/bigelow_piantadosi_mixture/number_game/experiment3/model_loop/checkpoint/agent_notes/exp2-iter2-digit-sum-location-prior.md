---
name: exp2-iter2-digit-sum-location-prior
description: Exp2 iter-2 cand_4 repair — open-ended graft was dup of cand_3 (RMSE 0.0003); resubmitted lo-bound location prior on digit_sum_concepts, smoke -1927.6
metadata:
  type: project
---
Exp2 iter 2, candidate_4 (2026-10-30), a refinement slot on digit_sum_concepts. The first attempt (open-ended range discount) was rejected as a near-duplicate of candidate_3's digit_sum_open_ended_range_discount. Candidates 3, 4 and 5 all picked the same graft.

Repair: grafted range_location_gradient_prior's beta*(lo-1)/99 interval prior onto digit_sum_concepts, using the Taylor-moment trick. Named `digit_sum_range_location_prior`. Smoke loglik -1927.6, 7 s per 200 iterations. Novelty vs the incumbent was not measured; the analogous change was RMSE 0.013 in iter 1.

**How to apply:** in refinement slots, check sibling candidates' model_name.txt first; parallel slots tend to pick the same graft. See [[exp2-iter1-range-location-prior]].

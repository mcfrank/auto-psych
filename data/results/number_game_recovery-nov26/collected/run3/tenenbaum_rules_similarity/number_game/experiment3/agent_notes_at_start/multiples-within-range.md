---
name: multiples-within-range
description: Round-4 repair (iter_4/candidate_4_repair_1): "multiples of m within a range" kind on parity_within_range base — weight 0.04, ll -1.4 vs incumbent
metadata:
  type: project
---
Round 4 (2026-10-30), iter_4/candidate_4_repair_1, multiples_within_range_concepts: incumbent parity_within_range_concepts + a 4th kind "multiples of m (3..9,11,12) within a range", same Erlang width prior. Quick 2x200 NUTS: kind weight ≈0.04; ll at posterior-mean p -1066.4 vs incumbent -1065.0 (no gain). p_yes RMSE vs incumbent on the data stimuli 0.012, so it clears the novelty gate. Biggest shifts: 60,54->58 (0.71→0.59), 39,60,12,45->99 (0.70→0.60).
The original candidate_4 was rejected because it copied gamma_width_parity_range_concepts (RMSE 0).
**Why:** yet another concept-space addition that gives nothing, same as [[concept-space-additions-map]].
**How to apply:** don't build more rule∩range conjunction kinds. Check novelty against sibling candidates in the same round before submitting.

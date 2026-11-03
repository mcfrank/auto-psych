---
name: digit-position-exemplar-retrieval
description: exp2 iter_4 c2 similarity slot: graded exemplar retrieval with digit-position (Hamming) metric; r→0.0017, nearly crisp; smoke OK, fast
metadata:
  type: project
---
exp2 iter_4/candidate_2 (2026-11-01), digit_position_exemplar_retrieval. It is graded_exemplar_concept_retrieval with s = r^(number of differing digit positions, 3-digit padding) in place of number-line distance. MAX_D=3, so the rule and range count tables are tiny. Range counts come from prefix sums per distance.
Results: smoke ll -2037.8; MAP ll -2033.5 with r≈0.0017, alpha .93, lambda .30. The digit gradient collapses to crisp, so the model risks being a near-duplicate of the crisp concept models. A check takes 2.4 min and 200 iters take 9 s.
**How to apply:** digit-wise similarity of targets to concept members is not supported. See [[graded-exemplar-uncertain-reliability]], [[digit-slip-stray-refinement]].

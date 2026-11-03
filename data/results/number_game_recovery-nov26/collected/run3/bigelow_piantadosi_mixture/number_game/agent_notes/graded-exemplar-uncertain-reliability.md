---
name: graded-exemplar-uncertain-reliability
description: exp2 iter_2 exemplar slot: number-line exemplar retrieval with exact Beta-marginal reliability (e_j of graded weights); r≈0.015; speed tricks
metadata:
  type: project
---
exp2 iter_2 candidate_0 (graded_exemplar_concept_retrieval, 2026-11-01). It revives exemplar_retrieved_number_concepts and adds the family-only prior and uncertain reliability. The Beta-marginal stays exact for graded similarity weights: sum_j c_j e_j(100*s_i). The range e_j enter linearly, so I tabulate log sum_h e_j per set and per cell on a logit(r) grid.
Result: smoke ll -2034.9, MAP r≈0.015 (almost crisp, so the novelty RMSE vs the incumbent may be small). Novelty and the full fit were still pending when this was written.
Speed: the first version ran at 7.7 ms/grad with 76 steps at 0.99, about 45 min. Two changes brought it to 2.5 ms/grad, 21 steps and about 9 min:
- Rule weights come from precomputed member counts at distance d ≤ 14, dotted with r^d. This replaced a gather or one-hot dot over the (100,45) similarity matrix.
- A flat 1-D gather for set_endorse[cell].
- The grid is narrowed to logit(r) ∈ [-8, -1.5], with prior N(-4, 1).
**How to apply:** for any graded-kernel concept model, see [[grid-tabulated-range-evidence]] and [[fast-incumbent-sampling]].

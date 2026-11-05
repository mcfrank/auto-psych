---
name: last-example-digit-recency
description: "Ends like the LAST example" lingering property gave +25.6 MAP loglik on final_digit_overshadowing_overlap (experiment2); first-example +7.6, middle 0
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_0/candidate_2, last_example_digit_recency_overlap). MAP log-lik on experiment2 data, incumbent final_digit_overshadowing_overlap = -3197.1. I added "ends in the last example's final digit" (only when the examples do not all share a digit) with salience recency*sal[final digit]:
- last example: -3171.5 (+25.6, recency about 0.84); any example: +14.5; first example: +7.6; middle examples only: 0.
- All of the last example's unshared properties at a discount: +4.3; first example's: -3.
**Why:** it is a real order (recency) effect, specific to the final digit. It addresses the critique's last_vs_first_example_digit statistic.
**How to apply:** more order effects may be in digit and surface features rather than arithmetic ones. See [[final-digit-overshadowing]].

Update 2026-11-02 (experiment2 iter_0/candidate_5_repair_1, seed_overshadowing_last_digit_recency): I combined three things: the seed family, digit overshadowing and last-example digit recency. The smoke-fit log-lik was -3159.5 (1 chain, 100 draws), and the fit took about 10 s. The first attempt at this slot re-implemented seed_multiple_digit_overshadowing_overlap exactly. The novelty gate rejected it (RMSE 0). Check the live models/ dir for duplicates before submitting. See [[seed-plus-digit-overshadowing]].

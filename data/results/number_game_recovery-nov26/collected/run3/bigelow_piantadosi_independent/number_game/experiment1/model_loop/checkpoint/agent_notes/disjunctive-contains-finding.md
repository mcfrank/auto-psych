---
name: disjunctive-contains-finding
description: Pricing "contains d" as a 3-primitive disjunction (no new param) on the multiplicative incumbent gained ~15 nats MAP; log-width Erlang intervals lost 2.4
metadata:
  type: project
---
Round 4 repair (2026-10-30), cell_1/experiment1, iter_4/candidate_4_repair_1. MAP screens on `multiplicative_not_exponential_concepts` (base -825.1):
- contains-d rules cost 3 primitives instead of 2: -810.2, with no extra parameter. The fitted contains-cost model gets -809.1. MAP p_yes RMSE between the two is 0.012 over the set x target grid, which is above the 0.002 novelty gate. Submitted as `place_value_disjunctive_contains`. Smoke-fit loglik was -812.8.
- An Erlang prior on log-ratio interval width (a Weber-style number line, aimed at the upward-generalisation critique): -827.6, which is worse.
The previous attempt in this slot (ends-in bonus) was rejected as an exact duplicate of final_digit_multiplicative_concepts.

**Why:** the ends-bonus and contains-cost slots are already taken by live models.
**How to apply:** check what's live before submitting a digit tweak. Screen script: iter_4/candidate_4_repair_1/scratch/screen2.py. See [[contains-digit-cost-on-incumbent]] and [[ending-bonus-on-incumbent]].

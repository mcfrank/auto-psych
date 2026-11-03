---
name: ends-bonus-plus-exponent-cost
description: Putting the ends-in-d bonus and the exponent cost together reached MAP loglik -799.5, about 27 nats better than either alone; lapse fits to 0
metadata:
  type: project
---
Round 4 (2026-10-30), cell_1/experiment1 MAP screens on final_digit_salient_concept_search (base -835.4):
- ends bonus + exponent cost: -799.5. Bonus about 1.8, exponent cost about 4.4, and lapse goes to about 0. Submitted as `final_digit_salient_no_exponent_rules` (iter_4/candidate_5). The smoke-fit loglik was -802.8.
- ends bonus + contains-digit bonus: -826.9. This is the critique's contains-digit over-use, but it gains less.
The incumbent `multiplicative_not_exponential_concepts` is at about -826.3 MAP. See [[final-digit-salience-finding]] and [[exponent-concept-cost-finding]].

**Why:** the two effects add up, which makes this the strongest concept-prior screen so far.
**How to apply:** an incumbent-slot candidate that adds an ends bonus would duplicate this model. For further gains, screen a contains-digit cost on top of this model.

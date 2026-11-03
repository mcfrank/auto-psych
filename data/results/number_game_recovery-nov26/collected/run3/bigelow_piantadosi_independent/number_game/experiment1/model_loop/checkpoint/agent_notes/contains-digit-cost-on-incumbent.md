---
name: contains-digit-cost-on-incumbent
description: Fitted extra cost on "contains digit d" rules, added to multiplicative_not_exponential_concepts, gained ~16 nats at max likelihood (cost ~1.5)
metadata:
  type: project
---
Round 4 (2026-10-30), cell_1/experiment1, candidate_2. On the incumbent `multiplicative_not_exponential_concepts`, the max-likelihood log-likelihood was -825.1. Adding a fitted cost on the 10 contains-d rules raised it to -809.1, with the cost at about 1.5. On the older erlang base, the same change gained only +2.6 (see [[incumbent-null-changes-round3]]). This addresses the iter_4 critique that people use contains-d less than the model predicts (0.43 observed vs 0.61). Submitted as `place_value_digit_search_cost`. Smoke-fit log-likelihood: -812.

**Why:** a change that fits to ~0 on one base can matter on another base.
**How to apply:** re-screen earlier null tweaks whenever the incumbent changes. The Nelder-Mead numpy screen lives at iter_4/candidate_2/scratch/screen.py. See [[exponent-concept-cost-finding]].

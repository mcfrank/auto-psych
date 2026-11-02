---
name: iter4-process-screens
description: Round 4 cand_2 MAP screens on WOR incumbent (NLL 902.77) — generative-typicality judgment -40, fixed rule/range arbitration -117, others tie; submitted description_length_rule_cost
metadata:
  type: project
---
Round 4, candidate_2 (2026-10-30). Numpy MAP screens on `concepts_sampled_without_replacement` (NLL 902.77):
- Answering "yes" by the generative probability p(target|D) instead of membership (sigmoid of the log ratio to chance): 943.5. Ruled out.
- A fixed rule-vs-range attention share instead of the Bayesian evidence share: 1020.1. Ruled out. Tempering the evidence ratio: t≈1, a tie.
- Rule share lambda varying with the number of examples: slope ≈0, a tie.
- Description-length complexity cost on the rule prior (lengths 1/2/4): 902.75 with gamma kept. Replacing gamma, it gave 903.45, with RMSE 0.0025 vs equal-prior WOR (close to the novelty gate).

Submitted the gamma-replacing version as `description_length_rule_cost` (smoke loglik -905.2, about 3 s per 200 iterations). The screen scripts are in iter_4/candidate_2/scratch/.

**Why:** at the ceiling, as in [[number-game-ceiling]] and [[iter3-process-screens]].
**How to apply:** don't re-screen these. Evidence-based arbitration between rules and ranges is strongly supported.

---
name: round-number-unrepresentativeness
description: exp3 iter2 cand1 — round-target log-odds penalty (no round examples) +5 loglik MAP on set_size_evidence_privileged_fives; numpy reimpl + other nulls
metadata:
  type: project
---
2026-10-31, exp3 iter_2/candidate_1 (`round_number_unrepresentativeness`). Base set_size_evidence_privileged_fives MAP loglik -3423.2 (logpost -3465.1).
- A penalty on round targets when no example is round (round_penalty ≈0.38) gives loglik -3418.2. Scaling it by n_examples: -3419.2. A symmetric round mismatch: -3421.9. Final-digit novelty: 0. 
- Decade blocks (20-29) instead of chart rows (21-30) tie (loglik -3423.4). A free rule size-principle exponent fits ≈0.98 and is null.
- Biggest per-set misses: 3,33,99 (38 nats), 8,64 (29: "even" over-extended), 83,38,58,68, and 14,98,91 (people reject multiples of 7).
- Gotcha: pytensor.function(m.value_vars, p_yes) after pm.set_data gave misaligned p. Use the exact numpy reimplementation in iter_2/candidate_1/scratch/w/post.py and resid.py. Its loglik matches the model.

**Why:** the round critique (q=0.064) is worth only ~5 nats. Bigger residuals are rule over-extension on specific sets.
**How to apply:** reuse scratch/w/screen.py and mkvar.py (a pair-feature logit shift) for quick screens. Related: [[exp3-iter2-critique-stats]], [[concept-space-saturated-exp3]].

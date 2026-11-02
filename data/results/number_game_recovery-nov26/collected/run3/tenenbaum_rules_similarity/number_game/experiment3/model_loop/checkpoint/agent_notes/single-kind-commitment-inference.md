---
name: single-kind-commitment-inference
description: exp3 iter_1 c0: Murphy&Ross single-category inference — kind posterior raised to exponent before mixing within-kind membership; smoke ll -3653.9
metadata:
  type: project
---
exp3 iter_1/candidate_0 (2026-10-31), `single_kind_commitment_inference`: beta-share learner; posterior over 9 kinds (8 rule families + range) raised to exp(log_commitment) (Normal(0,1)) and renormalised, then mixed with within-kind membership. Smoke check OK (~25 s per 200 draws), ll -3653.9 (same as incumbent). Commitment posterior not inspected — novelty risk vs beta_share if exponent≈1.
Trial-context mechanisms (carryover from previous trial) can't pass novelty: novelty_pool.json has only number_set/target, no trial_index.
**How to apply:** if this is rejected as a dup, kind-level mixing distortions collapse to normative on exp3. See [[kind-uncertainty-shifted-criterion]].

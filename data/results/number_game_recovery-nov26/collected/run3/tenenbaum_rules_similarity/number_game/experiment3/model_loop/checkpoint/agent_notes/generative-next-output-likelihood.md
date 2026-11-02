---
name: generative-next-output-likelihood
description: exp2 iter_4 c0: yes = sigmoid(a + b·log(100·P(t|X))) on beta-share learner; smoke ll −2511 (≈−240 vs incumbent)
metadata:
  type: project
---
exp2 iter_4/candidate_0 (2026-10-31), exploratory high-variance slot, submitted as generative_next_output_likelihood. Built on beta_share_width_exception_concepts code: predictive P(t|X) = Σ post(h)·(0.9·[t∈h]/|h| + 0.001), bias~N(0,2), sensitivity~HalfNormal(2). Re-tests the experiment1 generative_next_output_judgment idea, which crashed at fit (Hessian) and was never actually tested.
Smoke check OK, 12 s per 200 iterations; smoke log-lik −2511, so it loses by about 240 nats. Membership, not output probability, drives answers.
**Why:** a clean negative result for the generative reading of the question.
**How to apply:** don't retry generative or size-diluted readouts; see [[experiment2-decision-rule-nulls]].

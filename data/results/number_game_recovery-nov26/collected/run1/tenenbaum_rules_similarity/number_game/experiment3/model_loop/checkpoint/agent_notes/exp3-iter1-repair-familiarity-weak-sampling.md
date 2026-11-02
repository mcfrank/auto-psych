---
name: exp3-iter1-repair-familiarity-weak-sampling
description: exp3 iter_1 cand_0 repair (2026-11-01) — target-familiarity logit bias dead (-0.02); weak-sampling hypothesis elimination MAP -4009 (~495 nats worse), submitted as exploratory
metadata:
  type: project
---
- Previous cand_0 (target_as_evidence_positive_test) rejected as too slow: weight pins at 0 under HalfNormal, extra 5120x100 boost matrix per step.
- Familiarity bias (logit += b * centred -log(target)) on graded digit-sum incumbent: b = -0.022, no gain -> near-duplicate risk, abandoned.
- Weak sampling (example lik = 0.95 if consistent else 0.05, no size principle), same families + Erlang: MAP LL -4009 vs incumbent ~-3514; families: special .20, ends .47, ranges .28, parity ~0; scale 11. Fast (constant set_log_lik precomputed). Submitted as weak_sampling_hypothesis_elimination.
- Speed: on the loaded box the incumbent-style graph is ~450 ms/grad; under weak sampling, pre-summing ranges by width in prepare_observed gives 28 ms/grad (smoke fit 82 s, check OK).
**How to apply:** don't revisit familiarity/target-side biases; size principle is strongly supported (~500 nats). See [[exp3-iter1-process-probes]].

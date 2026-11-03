---
name: serial-position-reliability
description: experiment2 round 0: separate sampling-trust rate for first (or last) example ≈ 0.9, same as others; hard "first example never noise" loses ~600 nats
metadata:
  type: project
---
experiment2 iter_0/candidate_2 (2026-10-30), first_example_trust_concepts = gamma_width_exception_family_concepts with the first example's from-concept probability a1 inferred (later examples fixed 0.9). Quick 2x400 NUTS: a1 = 0.895 ± 0.011, LOO -2279.4 vs incumbent -2278.0; p_yes RMSE vs incumbent 0.0025 (borderline for the 0.002 novelty gate). Last-example version: a1 = 0.886, LOO -2279.0.
Parameter-free "first example must be in the concept" gave smoke ll -2908 vs -2273: it predicts yes ≈ 1 when the target is the first example, but people say yes only ~62% of the time there.
Only one permuted set pair (8,16 / 16,8) is in the data, so order effects are barely identifiable.
**Why:** serial-position mechanisms are now null twice, here and in experiment1 (serial_position_weighted_evidence, win_stay_lose_shift).
**How to apply:** don't spend order-effect slots on reliability or weighting by position. Linear-space A/B decomposition (likelihood = a1*A + (1-a1)*B) keeps per-example noise rates fast; see [[example-recall-subset-mixture]]. Test p_yes with model.replace_rvs_by_values, or compile_fn silently draws the RVs at random.

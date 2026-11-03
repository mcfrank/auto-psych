---
name: experiment2-round0-readout-and-sampling
description: exp2 iter_0/candidate_1 — log-odds report distortion is null (gain≈1, bias≈0); without-replacement sampling fits 4.4 nats worse; novelty-check scripting pitfall
metadata:
  type: project
---
Experiment2 round 0 (2026-10-30), iter_0/candidate_1, built on gamma_width_exception_family_concepts (quick 2x300 NUTS mean ll -2273.6):
- Linear-in-log-odds report distortion (gain, bias): ll -2274.2, gain 0.98, bias -0.05. Null; dropped (it would have been a near-duplicate).
- Submitted without_replacement_sampling_concepts (k-th in-concept example is one of |h|-(k-1) unseen members, no new params): ll -2278.0 (4.4 worse). p_yes RMSE vs the incumbent on the novelty pool is 0.0089, so it passes novelty.
- Participants 0-4 vs 5-9 see mostly different stimuli (only 32 shared pairs); yes rates 0.40 vs 0.58.
- Pitfall: to compare p_yes at fixed parameters, use graph_replace on the free RVs. The incumbent has 8 families, so a wrong simplex length gives bogus RMSEs.
**Why:** the readout is already calibrated, and sampling with replacement beats sampling without it.
**How to apply:** don't retry response-distortion readouts or sampling-assumption tweaks; see [[round4-null-refinements]], [[experiment2-data-structure]].

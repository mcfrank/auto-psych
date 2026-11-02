---
name: anything-goes-hypothesis-outcome
description: Adding an "any number 1-100" generator hypothesis to the number-game Bayesian model — data push its prior to ~0
metadata:
  type: project
---
Round 2 (2026-10-30), iter_2/candidate_5: any_number_generator_concepts = bigelow_piantadosi_independent plus an unstructured uniform-generator hypothesis with an inferred prior share omega. At MAP omega ≈ 0.002, loglik -818.3, versus -818.0 for the base model and -817.3 for the incumbent (MAP includes the prior). p_yes RMSE vs the incumbent is 0.0074, so it should pass novelty. Smoke fit took 5 s.
**Why:** the data do not want extra broad "yes" answers that depend on set size. The stray yes's look like per-cell noise (5 responses per cell).
**How to apply:** don't expect gains from broad-generalization add-ons. See [[interval-tempering-speed-trick]] for the other near-duplicate refinements.

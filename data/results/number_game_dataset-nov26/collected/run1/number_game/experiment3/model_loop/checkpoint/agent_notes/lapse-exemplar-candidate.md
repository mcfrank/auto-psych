---
name: lapse-exemplar-candidate
description: Round 2 decision-rule slot — submitted place_value exemplar + 50/50 lapse as `similarity_judgment_with_random_lapses`
metadata:
  type: project
---
Round 2, cell_1 (2026-11-01), iter_2 candidate_2: place_value_exemplar_similarity plus a coin-flip lapse rate (Beta(1.5,6) prior). check_candidate OK; smoke-fit log-lik -1595.6. Earlier screen put lapse at about 11 nats over the base exemplar.

**Why:** the brief asked for a decision-rule mechanism on the simplest inference. **How to apply:** if it does well, a later refinement could add the lapse to variability_tuned_exemplar_attention. Related: [[exemplar-variant-mle-screen]].

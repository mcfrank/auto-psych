---
name: adaptive-attention-exemplar-screen
description: Round 1 MLE screen — exemplar model with attention tuned by example variability beats incumbent exemplar by ~27 nats in-sample
metadata:
  type: project
---
Round 1, cell_1 (2026-11-01): in-sample NLL on 2520 responses: place_value_exemplar_similarity base 1602; + magnitude gradient exp(-g*range/100) 1590; + digit penalty exp(g*modal-final-digit share) 1590; both 1575 (g_range≈6.4, g_digit≈0.6). Submitted "both" as candidate_2 `variability_tuned_exemplar_attention`; check_candidate OK.

**Why:** the critique flagged that yes rates vary more between sets than the model predicts; this targets that. **How to apply:** if it gets admitted, later refinements could swap the exp-in-range form for distance/range normalization. Related: [[mixture-variant-mle-screen]].

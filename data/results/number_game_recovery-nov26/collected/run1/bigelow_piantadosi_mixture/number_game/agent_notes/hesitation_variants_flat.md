---
name: hesitation-variants-flat
description: MAP screens (iter_4 c4 repair, 2026-11-01) of span-hesitation variants on span_extrapolation_hesitant_guess_2 all fit within ~0.5 nats; novelty RMSE is the binding constraint
metadata:
  type: project
---
At MAP the incumbent span_extrapolation_hesitant_guess_2 has loglik -3110.5, stay≈0.055 (barely sticky), and reach≈exp(0.46) (weak hesitation). Variants: directional reach was rejected as a duplicate (RMSE 0.0012). Spread-scaled reach gave RMSE 0.0007 and n-scaled reach 0.0014, both too low. Fitted falloff power gave 0.0018 and a probability-weighting exponent fitted 1.0, so neither changes predictions enough. Log-scale distance cost 2.4 nats. Range-only hesitation cost 2.2 nats (RMSE 0.0049). Rule-only hesitation had loglik -3110.45 and RMSE 0.0026, and was submitted as rule_only_span_extrapolation_guess.

**Why:** the admission novelty threshold is p_yes RMSE 0.002 on model_loop/novelty_pool.json; fitted extensions with a null value collapse to the incumbent.
**How to apply:** screen candidates for RMSE against the incumbent on the real novelty_pool.json (scratch/rmse.py in iter_4/candidate_4_repair_1) before building; prefer structural changes over fitted nulls. See [[concept-space-saturated]].

---
name: exemplar-variant-mle-screen
description: Round 1 MLE screen of one-change variants of place_value_exemplar_similarity — parity dimension and lapse help most
metadata:
  type: project
---
Round 1, cell_1 (2026-11-01): in-sample NLL on 2520 responses: base 1602.1, parity-mismatch dimension 1589.3, 50/50 lapse 1590.6, set-GCD non-divisor penalty 1597.0, gcd(t,e)<2 penalty 1593.8, mean-not-sum 1602.4, gaussian/power/log distance ~1601-1603, tens-digit dim no help. Parity model has a flat ridge (similarity scales trade off with beta; logsumexp→max limit), priors pin it. Submitted parity as candidate_3 `parity_place_value_exemplar_similarity`.

**Why:** saves re-screening. **How to apply:** next refinement could try lapse on top of parity. See [[mixture-variant-mle-screen]].

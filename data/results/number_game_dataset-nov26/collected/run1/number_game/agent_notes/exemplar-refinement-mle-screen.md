---
name: exemplar-refinement-mle-screen
description: Offline MLE screen (round 1) of one-change variants of place_value_exemplar_similarity — digit-agreement attention helps most
metadata:
  type: project
---
Round 1, cell_1 (2026-11-01), in-sample NLL on 2520 responses: base 1602; final-digit mismatch penalty scaled by exp(gamma * the examples' pairwise final-digit agreement) 1586 (gamma≈0.55, a well-behaved regime); additive version 1587; parity-mismatch dimension 1589; lapse 1591; set-range-scaled magnitude gradient 1598; GCD-divisibility bonus 1599; set-size normalisation 1600. Submitted the digit-agreement variant as candidate_4 `digit_consistency_attention_exemplar`. Residuals: sets where every example shares a final digit (46,76,66; 78,48,88) get far fewer yes answers than the base predicts.

**Why:** saves re-screening. **How to apply:** next, try a parity dimension or lapse on top of digit attention. Related: [[mixture-variant-mle-screen]]

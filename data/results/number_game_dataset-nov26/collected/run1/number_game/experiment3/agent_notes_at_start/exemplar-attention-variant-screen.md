---
name: exemplar-attention-variant-screen
description: Round 1 offline MLE screen of variants of place_value_exemplar_similarity — digit attention driven by example final-digit agreement helps ~15 nats
metadata:
  type: project
---
Round 1, cell_1 (2026-11-01), in-sample NLL on 2520 responses: base place_value exemplar 1602; digit penalty + c_shift*(fraction of example pairs sharing final digit) 1587 (+1 param); also scaling the magnitude gradient by exp(-k*sd of examples) 1581 (+2); normalized GCM attention weight sigmoid(w0+w1*agree) 1598 (worse). Submitted the +1 param version as candidate_0 `coherence_driven_digit_attention`.
Data pattern: sets that share a final digit (15,85,45; 78,48,88; 63,23,43) have very low yes rates; sets with mixed final digits have high ones. This is what drives the critique's between-set spread.

**Why:** saves re-screening. **How to apply:** a later refinement could add dispersion-scaled magnitude breadth. Related: [[mixture-variant-mle-screen]]

---
name: parity-refinement-screen
description: Round 2 MLE screen of parity exemplar variants — adaptive parity attention useless; adding digit-consistency attention gives -18 nats
metadata:
  type: project
---
Round 2, cell_1 (2026-11-01), in-sample NLL on 2520 responses: parity_place_value base 1589.3; parity penalty scaled by exp(g*parity homogeneity) 1588.7 (no help); parity gated by homogeneity 1589.7; + digit-homogeneity-tuned digit attention 1571.1 (incumbent variability_tuned ~1575); both 1570.9. Submitted parity + digit-agreement attention as iter_2 candidate_5 `parity_space_digit_tuned_exemplar`; check OK (smoke loglik -1573).
Fits sit at the logsumexp→max ridge (beta≈0.002, c's large); priors pin it.

**Why:** the critique flagged a parity contrast for far targets in same-parity sets. **How to apply:** parity helps as a fixed dimension, not as adaptive attention. Related: [[exemplar-variant-mle-screen]], [[adaptive-attention-exemplar-screen]].

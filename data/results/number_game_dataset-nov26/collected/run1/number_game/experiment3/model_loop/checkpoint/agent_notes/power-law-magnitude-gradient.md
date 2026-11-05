---
name: power-law-magnitude-gradient
description: exp3 iter3 c1: gap**q magnitude gradient on parity_preserving_decisive_route, q≈0.46 at MAP, +7.9 nats MAP; barely moves polarization stats
metadata:
  type: project
---
Candidate power_law_magnitude_gradient (2026-11-02): incumbent with magnitude dissimilarity = c_mag·(dist+1e-3)^exp(log_shape). MAP loglik -4058.1 vs incumbent -4066.0; log_shape≈-0.77 (concave, like the pruned log-distance idea). Smoke -4065.8, 15 params, ~24 s.
Polarization PPC at MAP: single-example 0.167 vs incumbent 0.159 (obs 0.248); incoherent multi unchanged 0.113 (obs 0.187). So gradient shape does not fix the polarization critique.
**Why:** follows [[exp3-iter3-critique-stats]], [[log-distance-gradient-exemplar]].
**How to apply:** don't expect magnitude-shape tweaks to close cell-polarization gaps; they likely need cell/item-level heterogeneity.

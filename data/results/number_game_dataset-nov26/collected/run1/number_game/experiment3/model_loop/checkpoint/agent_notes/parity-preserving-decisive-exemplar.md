---
name: parity-preserving-decisive-exemplar
description: exp3 iter2 c3: parity-preserving multiples graft on coincidence_decisive incumbent, +9.9 MAP; decision-slope variants for incoherent polarization ~0
metadata:
  type: project
---
Submitted exp3 iter_2 candidate_3 `parity_preserving_decisive_exemplar` (2026-11-02): the incumbent [[coincidence-decisive-exemplar]] plus the multiples route × exp(-c_par_route·par_diff), taken from [[parity-preserving-multiple-route]]. MAP loglik went from -4077.6 to -4067.8 (single sets +4.8, coherent +6), and the smoke fit gave -4075.0.
One-parameter MAP screens on the incumbent, each aimed at the critique that incoherent and single-set cells are more polarized than predicted:
- slope × exp(g·log n): +0.7
- slope × exp(g·(1-spread)): +1.8
- graded partial coincidence on the slope: +2.6
- separate slope for single examples: +3.8, but g<0 (less decisive)
- slope tied to the best example's similarity: 0
- magnitude distance measured to the examples' span (hull): -2.4, plus a mixed version ~0
- Gaussian distance: -15; power-exponent distance: ~0
None of them moved the expected polarization (about 0.14 against the observed 0.216).
Incumbent MAP: magnitude is nearly flat (c_mag≈0.06 per 10 units). Incoherent-set predictions are bunched at 0.3-0.5 and far targets are over-endorsed (0.36 predicted vs 0.24 observed).
**Why:** the polarization gap probably needs item-level features the model doesn't have, not a change to the decision rule.
**How to apply:** don't retry decision-slope fixes for incoherent polarization; look at which far targets in incoherent sets get zero yes answers.

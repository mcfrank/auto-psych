---
name: arithmetic-verdict-decisive-exemplar
description: exp3 iter2 c0: steeper response slope for multiple/factor-kin targets on coincidence_decisive, +6.3 nats MAP; barely moves incoherent polarization stat
metadata:
  type: project
---
exp3 iter2 candidate_0 `arithmetic_verdict_decisive_exemplar` (2026-11-02). It adds slope × exp(verdict_gain·arith_related) to coincidence_decisive_exemplar, where arith_related means the target is a multiple or factor-kin of some example. MAP -4071.4 vs incumbent -4077.6. Smoke -4078.7.
Diagnosis of the incoherent-polarization critique: in incoherent and single-example sets the incumbent never predicts below 0.15. The excess polar cells are mostly lone-example multiples (76 cells: polar 0.30 vs null 0.13). Odd examples' even multiples are rejected (17→34, 43→86, 49→98); that parity-switch pattern is what parity_preserving_multiple_route captures, so a refinement slot should graft it onto the incumbent.
Screened with no gain: set-size response slope (+0.7), lone-example parity weight (0), lone parity_homog=1 (-2.5), adding round or same-digit to the verdict gate (no better).
**Why:** decisiveness gated on the arithmetic route helps only a little; the item-level mispredictions come from parity switching.
**How to apply:** try parity_preserving + coincidence_decisive in a refinement slot. MAP screen script: /tmp/c0s/fit.py (may be gone). Related: [[coincidence-decisive-exemplar]], [[parity-preserving-multiple-route]], [[exp3-iter2-critique-stats]].

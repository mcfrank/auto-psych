---
name: arithmetic-relation-decisive
description: exp3 iter2 c2: arithmetic-relation-gated response slope on coincidence_decisive incumbent, +6 MAP, +15 leave-set-out CV; per-number target appeal overfits
metadata:
  type: project
---
exp3 iter2 candidate_2 `arithmetic_relation_decisive`: one term added to the incumbent's slope exponent, beta*exp(sharpening*coinc + w*arith_rel). arith_rel = 1 if the target is a multiple of an example, shares gcd>=3 with one, or is a round target beside a 0/5 example. MAP -4071.5 vs -4077.6 for the incumbent. Five-fold leave-set-out CV: -4130.3 vs -4145.1. Smoke -4079.6. It only partly closes the incoherent-cell polarization critique: expected fraction 0.138→0.145, observed 0.216.
Screened on the same slope: lone-example gate +4 MAP (w<0), far-target gate +3 MAP / +3 CV, multiples-only gate +3 CV, gcd>=3-only gate +14 CV.
A per-number target appeal (100 random effects) gave +77 MAP but -87 leave-set-out CV. Even with sd fixed at 0.3 it was still -14 CV. Ruled out.
**Why:** in incoherent sets the incumbent predicts ~0.2-0.5 everywhere. Much of the polarization looks item-specific and does not carry over to new sets.
**How to apply:** don't retry target-level random effects. Related: [[coincidence-decisive-exemplar]], [[exp3-iter2-critique-stats]].

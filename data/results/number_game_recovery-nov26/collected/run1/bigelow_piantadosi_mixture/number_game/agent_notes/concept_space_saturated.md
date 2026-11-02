---
name: concept-space-saturated
description: MAP screens (iter_4, 2026-11-01) found no added concept family helps span_extrapolation_hesitant_guess_2
metadata:
  type: project
---
Added as a fitted-weight rule family to span_extrapolation_hesitant_guess_2, every family's weight was pushed toward zero by MAP, with no log-lik gain over the base (-3110.50). Families tested: offset progressions (steps 3-12), ±1 of squares/powers, multiples of 13-50, digit-sum classes, repdigits, triangular/Fibonacci/perfect powers/factorials, and "ends in a or b" unions. Progressions even cost about 2 nats at weight e^-6.7, because the size principle still gives them posterior mass on multi-example sets. Reweighting existing rule groups changed things by at most 0.6 nats.

**Why:** the remaining misfit is not in the concept space; residuals per cell look like n=5 noise.
**How to apply:** for concept-space briefs, run a quick MAP family screen (iter_4/candidate_0/scratch/map.py pattern, ~30 s each) before building; expect null results. See [[check-candidate-runtime]].

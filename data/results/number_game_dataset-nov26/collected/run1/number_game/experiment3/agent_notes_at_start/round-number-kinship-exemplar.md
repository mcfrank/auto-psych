---
name: round-number-kinship-exemplar
description: exp2 iter3 c0: multiple-of-10 target gets distance-free similarity route to examples ending 0/5, MAP +12.3 nats over factor_kin_multiple_route
metadata:
  type: project
---
Residual screen (2026-11-02) on incumbent [[factor-kin-multiple-route]]: multiple-of-10 targets under-predicted (0.43 obs vs 0.36) mainly when the set has a 0/5-ending example (0.46 vs 0.31). MAP log-lik: incumbent -2682.5; round route (t%10==0 & e%5==0) -2670.2; e%10 only -2675.9; t%5 & e%5 -2675.1; set-gated round logit bias -2671.8; 0/5 digit-family -2673.5; roundness-mismatch dimension 0 gain.
Times-table *product* gate (multiplier<=12) was worse than the example<=12 gate (-2681.7 vs -2677.7): multiplier size isn't the signal.
Prime-set prime targets: weak evidence (59,89,79 doesn't favour primes).
**Why:** new info (base-10 roundness) that no live model uses.
**How to apply:** if admitted, the round route could be grafted onto times_table gating in a refinement slot.

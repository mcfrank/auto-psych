---
name: exp3-r3-incumbent-residuals
description: Exp3 round 3 MAP tests of single changes to upward_count_start_boundary; which cues help and which do nothing
metadata:
  type: project
---
Exp3 round 3 (2026-10-18), incumbent upward_count_start_boundary MAP ll −4642.8. Fitted weights: size w≈0.69 per 100 units (weak), graded-factor w≈0.005 (factor similarity basically unused).
Changes tested, MAP ll gain each: isolation memory, where an example differing from the rest in parity and digit weighs more (+6.6, submitted as distinctive_example_isolation_memory); a tens-digit/decade similarity dimension (+5.6); factor-mismatch normalisation variants with max, geometric mean, or 1 having no factors (0 or worse); exp(-d^q) similarity shape (+0.3).
Residuals left over: targets 1 and 2 over-predicted by about 0.07; set-level yes-rate SD is 0.147 observed vs 0.114 predicted.
**Why:** saves re-testing dead ends. Gains this small (~5 nats) probably come out below 1 dse in ELPD.
**How to apply:** skip factor-similarity tweaks. The decade dimension is still untried as its own candidate. See [[residual-ascending-below-start]] for the workflow (scratch/resid.py in iter_3/candidate_4).

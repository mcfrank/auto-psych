---
name: divisor-family-exemplar
description: exp2 iter_1 c2 — off-divisor iff gcd(target, set gcd)<3, applied to single examples too; smoke -2711.9 vs incumbent -2718.6
metadata:
  type: project
---
Experiment 2, iter_1 candidate_2 (2026-11-02): `divisor_family_exemplar` refines `common_divisor_sharpened_exemplar` with one feature change. People consider "multiples of d" for every d>=3 dividing all the examples, so the penalty applies when gcd(t, g)<3 rather than when t%g!=0, and single examples count too (g = the example). No new parameters. Smoke log-lik -2711.9. It targets the single_example_shared_factor_contrast critique.
Data note: the single-example sets are sparse (12 sets). Almost all their targets share a factor >=3 with the example; only the set "7" contrasts multiples (0.78) with digit-7 targets (0.53).
**Why:** it also fixes "90, 45", where strict use of the GCD failed (30 and 60 are endorsed).
**How to apply:** compare its LOO with the incumbent's. See [[common-divisor-sharpened-exemplar]] and [[exp2-iter1-critique-stats]].

---
name: exp2-iter1-critique-stats
description: experiment2 iter_1 critique of common_divisor_sharpened_exemplar — the 8 PPC statistics proposed and why
metadata:
  type: project
---
experiment2 iter_1 critique (2026-11-02) of `common_divisor_sharpened_exemplar`. Stats (observed values): digit-match divisor exemption (gcd>=3 sets, digit-matching targets, multiples minus non-multiples, obs 0.04; e.g. 19/59 for "69,9,39,21" ~0.6), single-example shared-factor contrast (obs -0.10, model has no factor term for n=1), first-vs-last nearest example (0.05), set-size 3-4 vs 1-2 on background targets (-0.055), exact-example rate (0.82), digit-match distance corr (-0.02, flat), background trial drift corr (0.035), background per-set rate SD (0.20).
"Background" = no example final digit match and >=10 from every example.
**Why:** test whether the constant divisor penalty wrongly hits digit-matching targets, and the iter_0 leftovers (set breadth, ceiling, drift).
**How to apply:** results in experiment2/model_loop/iter_1/critique/critiques.md; see [[exp2-iter0-critique-stats]].

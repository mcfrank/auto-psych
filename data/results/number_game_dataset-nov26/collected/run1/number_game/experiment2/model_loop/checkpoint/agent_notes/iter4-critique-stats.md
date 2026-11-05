---
name: iter4-critique-stats
description: iter_4 critique of shared_factor_parity_exemplar — which 8 PPC statistics were proposed and why
metadata:
  type: project
---
iter_4 critique (2026-11-02) of `shared_factor_parity_exemplar` uses these statistics: set-level yes-rate SD (repeated from iter_3, where it survived FDR), non-matching-target rate in sets with >=3 same-final-digit examples, a powers-of-2 target contrast in all-power-of-2 sets, a multiples-of-5 contrast in all-multiple-of-5 sets, mean within-set item SD (sets with >=9 targets), the trial-level slope on set size, the trial_index drift correlation, and a shared-divisor-with-every-example vs coprime contrast among targets that match no example's final digit.
**Why:** they target arithmetic rules (powers of 2, multiples of 5, divisibility) and set-level rule strictness, which the similarity model can't express, plus nuisance effects (set size, drift).
**How to apply:** read iter_4/critique/critiques.md for results before reusing these; see [[iter3-critique-stats]], [[shared-factor-exemplar-screen]].

---
name: exp3-iter1-critique-stats
description: exp3 iter_1 PPC stats on fives incumbent; raw data shows times-table gap only 0.04, random sets in-range < out-of-range
metadata:
  type: project
---
2026-10-31, exp3 iter_1 critique of fives_privileged_decoupled_prime_odd. I wrote 8 stats. Observed values: times-table sharpness (multiples of m vs non-multiples, m≠5) is 0.04, nearly flat. For unstructured 3-4 example sets, in-range minus out-of-range is -0.08, so people don't use the range. Odd-set odd-vs-even gap 0.29. Shared final digit gap 0.88. Examples as targets 0.79 yes. Singleton near-far gap 0.07. Trial slope -0.03 per 100 trials. Within-pair variance 0.135 (pairs have ~5 reps).
**Why:** these are candidate discrepancies; check ppc_results.json for which ones are significant.
**How to apply:** if they're significant, times-table over-generalisation and interval over-use on random sets are targets. Related: [[concept-space-saturated-exp3]], [[repdigit-marked-odd-graft]].

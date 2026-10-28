---
name: limited-capacity-example-memory
description: exp3 iter_2 c0 (2nd attempt): incumbent + WM capacity 2/3/4+ keeps last example + random earlier ones; sparse entry trick for subset mixtures
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_0 was retried; the first attempt (ascending_sequence_range_bounds, [[ascending-sequence-range-bounds]]) left no files. Bottom-marking was already taken by c3/c4/c5 and parity-extrapolation by c2, so c0 became `limited_capacity_example_memory`: compressed_rule_extension evaluated on retained subsets (the last example is always kept, plus a random K-1 of the earlier ones), with capacity Dirichlet over {2,3,4+}. It targets set_size_near_outside_gap. In the raw data, yes to examples falls with set size (0.95, 0.92, 0.79, 0.76 for n=1-4).
Speed: there are 554 subset units and 4272 (pair, unit) rows. A dense gather took 0.148 s per gradient. Sparse nonzero entries (flat posterior index × value, then inc_subtensor) took 0.056 s, vs 0.045 s for the incumbent.
Check OK: smoke LL -3096.3 (incumbent about -3094.5), about 300 s per smoke sampling run, so the 30-min limit is at risk.
**How to apply:** check the outcome and the capacity posterior. Reuse the sparse-entry trick for any mixture over sub-sets. Related: [[random-example-forgetting]], [[pymc-model-speed-patterns]]

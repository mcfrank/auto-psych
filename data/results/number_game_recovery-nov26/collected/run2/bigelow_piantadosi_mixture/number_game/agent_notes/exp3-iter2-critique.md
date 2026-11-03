---
name: exp3-iter2-critique
description: Exp3 iter-2 critique of heading_weber_with_upward_counting — 8 new stats (squares, primes vs odd, round targets, shared digits, adjacency, AP next/mirror, magnitude breadth); iter-1 0/8 but doubling z=1.86, per-set SD z=-1.83
metadata:
  type: project
---
Iter-1 critique (2026-10-31) found 0/8 significant; the nearest misses were double/half yes (obs above null, z=1.86) and per-set yes-rate SD (obs below null, z=-1.83).
For iter 2 I wrote the files from a Python generator (/tmp/mk_stats.py, using the same header as iter 1). Bash `mk` quoting broke on apostrophes, so don't use it.
The 8 stats and their observed values: squares | all-square sets 0.475, prime minus odd-nonprime 0.19, far 1ex minus 4ex -0.001, round minus x1/x9 targets -0.009, shared-digit far +0.10, adjacent(±1) minus 3-5 +0.05, AP next minus mirror +0.12 (only 5 next-term rows, weak), single-example near large minus small +0.10.
See [[exp3-iter1-critique]], [[critique-stat-no-imports]].

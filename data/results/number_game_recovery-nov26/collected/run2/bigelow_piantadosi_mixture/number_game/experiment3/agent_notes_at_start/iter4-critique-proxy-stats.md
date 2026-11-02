---
name: iter4-critique-proxy-stats
description: Iter-4 critique trick — PPC stats can embed a fixed-param Bayesian proxy (stimulus-only, cached) to slice by predicted probability; no iter 0-3 stat was ever significant
metadata:
  type: project
---
Through iter 0-3 no PPC statistic reached p<=0.05. The near-misses were trial-index slope (p=0.05), cell unanimity (p=0.06) and in-set target yes (observed 0.68 vs 0.74). All 5 participants see the same trial order.

Iter 4 (2026-10-30) stats embed `_proxy(df)`: rules+intervals, gamma=0, lambda=0.5, WOR, no lapse. It depends only on the stimuli and is cached in a module dict, so it takes about 1 ms per call. The stats slice by its q: ambiguous band, confident members, late-vs-early false alarms, false-alarm cell overdispersion, single-example correlation. Three new relation stats: digit reversal, double/half, digit-sum.

**How to apply:** reuse the proxy template from iter_4/critique/test_stats; check ppc_results there before re-proposing. See [[iter3-noise-ceiling-wor]].

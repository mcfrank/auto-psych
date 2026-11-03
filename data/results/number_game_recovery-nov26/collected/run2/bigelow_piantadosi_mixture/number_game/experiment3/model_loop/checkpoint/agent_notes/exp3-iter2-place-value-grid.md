---
name: exp3-iter2-place-value-grid
description: Exp3 iter-2 cand_2 — tens×ones grid blocks replacing number-line ranges, smoke LL -3629.5 (~440 nats behind), submitted as loud high-variance test
metadata:
  type: project
---
Exp3 iter 2 cand_2 (2026-10-31, exploratory slot, brief asked for simpler/high-variance). Submitted `place_value_grid_blocks`: intervals replaced by blocks of the place-value grid (tens band × ones band, deduped, uniform prior), with rules, digit sums, WOR sampling, 10% strays and lapse kept. It has 4 params, all precomputed, and samples in ~1 s. Smoke LL -3629.5 vs ~-3190 for the incumbent, so continuous number-line ranges matter and decade-boundary structure does not replace them.
**How to apply:** don't retry grid-only magnitude concepts; any grid idea would need to be added to ranges, and family additions tend to collapse to dups ([[exp3-decade-stated-prior]]). See [[exp2-pure-rule-induction]].

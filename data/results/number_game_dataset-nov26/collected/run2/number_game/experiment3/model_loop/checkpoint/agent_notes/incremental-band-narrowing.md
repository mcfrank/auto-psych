---
name: incremental-band-narrowing
description: band width shrinking per extra example on exp2 progression_band_first_digit_primacy: +8.9 MAP; magnitude-order (near-last vs near-first) has no within-set effect
metadata:
  type: project
---
2026-11-02, exp2 iter_4 c1 (incremental_band_narrowing). Band width = 0.02 + band_width*exp(-narrowing*(n-1)) on top of the incumbent. MAP loglik -3117.3 vs -3126.2 (narrowing ~0.84). Smoke -3125.0, check OK, 23 s.
Raw data: for single-example sets the yes-rate stays flat (~0.58) with distance; for multi-example sets it drops with distance outside the range. Within a set, targets nearer the last example vs the first show no difference (mean +0.01 ± 0.04), so magnitude recency/primacy is a dead end. A "revision count makes people fall back on range" idea was confounded with set size.
**Why:** an order-themed brief aimed at the inrange_nonshared_rate critique.
**How to apply:** check whether it survives the full LOO. Width depending on set size may be what the single-example leniency/band models were partly capturing. See [[graded-magnitude-band]], [[progression-band-primacy]].

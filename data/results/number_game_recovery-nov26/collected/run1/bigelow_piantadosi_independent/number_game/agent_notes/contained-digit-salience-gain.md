---
name: contained-digit-salience-gain
description: MAP screen of number-game variants (2026-10-31, exp3 iter_3/cand_3) — separate kappa for "contains digit" rules gains ~20 nats; pooling them with final-digit salience loses ~60
metadata:
  type: project
---
On gamma_shaped_range_width_prior, MAP log-lik: incumbent -2556.6; contains-digit rules sharing the final-digit kappa -2615.5 (worse); separate kappa_c (fit 1.65) -2536.2. Submitted as salient_contained_digit_rules; check passed (51 s sampling).
**Why:** quick MAP screens (scratch/map.py in iter_3/candidate_3) are a cheap way to rank variants before a full check.
**How to apply:** a pm.find_MAP run per variant in parallel takes a few minutes; reuse that script. Related: [[interval-prefix-sum-speedup]].

Iter_4/candidate_3 MAP screen on top of salient_contained_digit_rules (base -2536.2): + range-plus-exception graft -2535.7 (+0.5, not worth it); + decade rules "tens digit d" (10-19..90-99) pooled with kappa_c -2533.4; with own kappa_t (fit 2.77) -2530.5 (+5.7). Submitted the latter as salient_decade_place_value_rules; check OK, 58 s sampling. Scratch variants in iter_4/candidate_3/scratch/v. Note `patch` fails writing in place here (owner error); use `patch -o out in`.

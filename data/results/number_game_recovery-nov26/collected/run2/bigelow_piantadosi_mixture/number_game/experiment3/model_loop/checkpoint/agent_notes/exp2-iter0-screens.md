---
name: exp2-iter0-screens
description: Exp2 iter-0 cand_3 MAP screens on WOR incumbent (NLL 1928.5): stray rate sharply peaked at 0.9, interval width prior +1.2 nats submitted
metadata:
  type: project
---
Experiment 2, round 0, candidate 3 (2026-10-30). MAP screens on `without_replacement_size_principle` (NLL 1928.5; lapse≈0.006 with guess_yes→1, i.e. a pure yes floor):
- Rule prior ∝ size^-g: 1928.1.
- Interval prior ∝ width^-g: 1927.3 (g≈0.12, prediction RMSE 0.0099, max 0.04).
- Fixed stray rate of 0.20 / 0.15 / 0.05 / 0.02: 2001.6 / 1952.1 / 2002.7 / 2264.6. So 0.9 is near-exact, and freeing it would be a near-duplicate.
- Evidence exponent: 1928.0 (≈0.98).

Residuals by distance and by set size are within 0.02. The largest per-set residual is 73,33,3 (6 yes vs 1.8 predicted).
I submitted `narrow_range_prior_without_replacement`, with intervals grouped into 100 width classes so that gamma only reweights them. Tabulating per (set, target) cell instead of per row cut the gradient from 20 ms to 2.9 ms; the smoke loglik was -1930.2. The screener is at experiment2/.../iter_0/candidate_3/scratch/s/.

**Why:** exp2 looks close to model-generated data, and the incumbent sits at its ceiling ([[number-game-ceiling]]).
**How to apply:** don't re-screen the stray rate or the evidence exponent on exp2.

---
name: exp3-iter0-weber-heading
description: Exp3 iter-0 cand_3 — heading incumbent at noise ceiling; all one-param screens ≤1.7 nats; Weber width prior +1.0 submitted; 2-D Taylor moment trick
metadata:
  type: project
---
Exp3 iter 0, candidate_3 (2026-10-31), refinement slot on `heading_direction_range_extrapolation`. Numpy MAP NLL is 3190.9; the cell-saturated ceiling is 2538 over 1307 cells, a gap of about 650, which is roughly the binomial noise.

MAP estimates on exp3: heading beta is about 0.04 (nearly null here), gamma is about 0, lapse goes to 0 and guess_yes to 1. ALPHA = 0.9 is a sharp optimum: 0.95 is +58 nats worse and 0.98 is +180 worse.

Screens (Δ nats):
- contains-digit weight, ends-in weight, rule tempering, rule size prior: ≤0.2
- open-ended: 0.4
- min-room: 1.2
- relative width (2 params): 1.35
- margin: 1.7 (duplicates margin_and_heading)

Submitted `heading_extrapolation_weber_width`: (w/mid)^gamma replaces w^gamma, with no extra parameter. It gains +1.0 nats, has a p_yes RMSE of 0.007 against the incumbent, and the smoke run gave -3193.2 in 30 s.

Implementation: the prior factor uses a 2-D Taylor expansion over the scaled log relative width (28 terms) and the heading room (24 terms), with gamma ~ Normal(0, 0.5).

The top per-set residual, z = 10 on (4,64,25,81), is an artefact of lapse going to 0 at MAP. It is not a mechanism.

Scratch screen harness: experiment3/model_loop/iter_0/candidate_3/scratch/screen.py.
Pitfall: `pkill -f screen.py` killed my own shell.
See [[number-game-ceiling]], [[exp2-iter0-relative-width-prior]].

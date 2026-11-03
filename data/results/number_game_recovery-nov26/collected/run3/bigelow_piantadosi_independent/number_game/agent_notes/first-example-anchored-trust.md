---
name: first-example-anchored-trust
description: exp3 iter3 first-example stray rate on round_landmark_veto +0.8 loglik (e1≈0.084, primacy trust); last-example null; fixed extremes −34..−55
metadata:
  type: project
---
2026-10-31, exp3 iter_3/candidate_0 (`first_example_anchored_trust`). On round_landmark_veto (MAP loglik -3415.3), I gave one example position its own fitted stray rate:
- First example: +0.8 loglik, e1≈0.084, below the later examples' 0.1. So on exp3 the first example is trusted more (primacy), the opposite of what exp2 showed.
- Last example: +0.3, e≈0.11.
- First example's rate fixed at 0.01 or 0.3: −34 and −40. Last example's fixed at 0.01: −55.
- Novelty RMSE against the incumbent at MAP is 0.005, which is above the 0.002 duplicate cut. The check passed (smoke loglik -3423.1, 158 s for 200 draws).
Implementation: interval groups get a first-in flag, so G is 1600. It is exact: with e1=0.1 the loglik equals the incumbent's.
Gotcha: novelty screens need `replace_rvs_by_values` before compile_fn. Without it, RMSE comes out around 0.45. Script: scratch/nov.py.

**Why:** order-effect briefs keep recurring. [[order-effects-on-prime-over-odd]] already shows they give about 1 nat or less.
**How to apply:** don't spend more screens on order effects. Related: [[position-stray-rate-finding]].

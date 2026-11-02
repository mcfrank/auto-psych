---
name: repdigit-marked-odd-graft
description: exp3 iter0 cand5 — repdigit rule grafted onto prime_salient_marked_odd (prime and odd decoupled) beats incumbent by +11.8 MAP
metadata:
  type: project
---
2026-10-31, exp3 iter_0/candidate_5 (`repdigit_marked_odd_prime_concepts`). On exp3, the incumbent repdigit_pattern_chart_concepts has MAP NLL 3465.8. Splitting its single prime_over_odd shift into a separate prime bonus (≈1.6) and odd cost (≈0.6) gives 3454.0, a gain of 11.8. Prediction RMSE against the incumbent is 0.026 on observed cells, max 0.105. Smoke loglik -3461.1, about 110 s per chain, and the check printed OK.
The incumbent's biggest exp3 misses: 3,33,99 (93 yes 5/5); 8,64 (people say no to multiples of 8); odd sets (63,27,99,33; 31,3,1,15; 1,8,27) where odd targets are under-predicted. 83,38,58,68 has 300 rows.
The scratch screen is iter_0/candidate_5/scratch/screen.py. It reuses the incumbent module's arrays with full interval enumeration and takes about 30 s per fit.

**Why:** on exp3 the data want more "odd" than the coupled swap allows.
**How to apply:** for later exp3 refinements, start from this decoupled base. Related: [[prime-over-odd-swap]], [[repdigit-concept-repair]].

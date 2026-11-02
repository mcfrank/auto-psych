---
name: exp2-iter0-range-occam
description: Exp2 iter-0 cand_5 MAP screens on WOR incumbent (NLL 1928.5); everything within 1.2 nats; range-only Occam prior submitted
metadata:
  type: project
---
Exp2 iter 0, candidate_5 (2026-10-30), refinement slot. MAP screens on the WOR incumbent (NLL 1928.49; MAP lapse 0.006, guess_yes about 1):
- rule breadth prior: 1928.1
- shared occam g (rules and ranges): 1928.5
- range-only power prior: 1927.3, g=0.12
- range exponential prior: 1927.3
- Erlang: 1931.3
- digit-family weight: 1928.5
- outlier rate: sharp optimum at 0.9 (0.85 gives 1952, 0.95 gives 1984), so a free outlier rate is pointless here.

Submitted `occam_range_width_prior` (refines pruned occam_prior_without_replacement, with the bias on ranges only). RMSE vs the incumbent is 0.010.
The biggest per-set residuals are 85,19,91 (odd overpredicted, -12) and 71,61,26,81 (+14). Single-example sets are overpredicted at distances 3-20.
Speed: gathering only the asked (set,target) cells (900 of 20300) cut 200 iters from 36 s to 7 s. Smoke loglik -1930.2.
Screener: experiment2/model_loop/iter_0/candidate_5/scratch/s/screen.py.

**How to apply:** exp2 is at the noise ceiling too. See [[number-game-ceiling]].

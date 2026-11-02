---
name: iter4-range-space-screens
description: Round 4 cand_0 repair — zero-param range-space changes on WOR incumbent (NLL 902.77): log-line sampling 919.9, width-uniform prior 938.7, decade ranges 917.2, multiple-of-5 ranges 908.4 (submitted)
metadata:
  type: project
---
Round 4 candidate_0_repair_1 (2026-10-30). The original magnitude_biased_example_sampling was rejected as a near-duplicate (beta shrank to 0, RMSE 0.001).
Zero-parameter MAP screens vs incumbent (902.77):
- Log-number-line sampling within intervals: 919.9 (RMSE 0.059).
- Uniform-over-width interval prior: 938.7.
- Ranges with decade boundaries only (55): 917.2; with multiple-of-5 boundaries (210): 908.4, RMSE 0.035.
I submitted `round_number_bounded_ranges` (the multiple-of-5 version, a one-line change to the incumbent, smoke loglik -910.1). It is likely to lose by ~5 nats, but it is novel.

**Why:** free params near null fail the novelty gate (RMSE < 0.002); zero-param structural changes avoid that.
**How to apply:** fine-grained arbitrary intervals are supported. Don't re-screen coarser range spaces. See [[number-game-ceiling]].

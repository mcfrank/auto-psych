---
name: size-inflation-softplus
description: Convergence fix for incumbent-family models — size_inflation is the worst-mixing param; softplus(Normal) beats HalfNormal/LogNormal
metadata:
  type: project
---
2026-10-21, experiment3 iter_3 candidate_2_repair_1 (adaptive_breadth_criterion): admission rejected for R-hat 1.054 / ESS 67. A 2-chain warm-start bench (250/250) showed size_inflation was the bottleneck (posterior ~2.3 ± 1.0, roughly symmetric on the linear scale, so skewed in log space). Bulk ESS: HalfNormal(10) 181, LogNormal(log 3, 0.75) 223, softplus(Normal(2.5, 1.5)) 734. After the change the min ESS over all params was 584, with no divergences.
Also centre/scale any set-level covariate (log breadth mean 0.71, sd 0.2 at MAP) so its slope does not trade off with bias.
**Why:** the warm-start fixed dense mass matrix assumes Gaussian geometry; skewed params mix badly.
**How to apply:** for convergence rejections on this base, bench with /tmp/c2r/bench.py (PYTHONPATH=. needed) and reparameterise size_inflation via softplus. See [[volatile-model-speed]], [[slow-fit-diagnosis]], [[adaptive-breadth-criterion]].

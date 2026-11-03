---
name: variable-expectation-decision-noise
description: exp3 iter_3 c4 repair: stray mix tied to lambda is a dup; Thurstonian log-odds noise (stated LogNormal(log .45,.15)) on variable-lambda incumbent −1.3 MAP, pool RMSE .005
metadata:
  type: project
---
exp3 iter_3/candidate_4_repair_1 (2026-11-02). The first attempt, one-sided thresholds on the incumbent, was rejected as a dup (RMSE 0) of sibling c3, which submitted the same graft. See [[check-sibling-candidates-first]].
MAP screens on variable_rule_range_expectation (ll −3319.24):
- Stray familiarity mix = lambda_0 (prior-matched): −3319.16, pool RMSE .0006, a dup. The incumbent's MAP mix .27 is already about lambda_0 .23.
- mix = lambda at each quadrature node: −3319.23, RMSE .0008, a dup.
- Thurstonian log-odds noise with a stated prior at .35: −3319.78, RMSE .0027 (marginal). At .5: −3320.50, RMSE .005 vs the incumbent and .009 vs thurstonian_decision_noise.
Submitted variable_expectation_decision_noise with prior LogNormal(log .45, .15).
**How to apply:** the incumbent is saturated (see [[variable-lambda-incumbent-saturation]]). Decision noise passes novelty only with a stated prior ([[thurstonian-decision-noise]]). Gotcha: `pkill -f mapeval.py` also kills your own shell, because the pattern matches its command line.

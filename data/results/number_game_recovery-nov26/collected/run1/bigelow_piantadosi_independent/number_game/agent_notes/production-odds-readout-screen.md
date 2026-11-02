---
name: production-odds-readout-screen
description: Readout "yes-odds grow with P(program produces target)/chance" (size principle on the target) loses ~143 nats MAP vs membership readout
metadata:
  type: project
---
2026-10-31, experiment3 iter_4/candidate_2 (exploratory slot): generative_production_odds_judgement = incumbent salient_contained_digit_rules learner, but p_yes = sigmoid(bias + slope*log(100*P(t|X))) with intervals weighted by an extra 1/W(a,b) prefix table (prefix2). MAP log-lik -2679.2 vs incumbent -2536.2 (gamma->0, omega->0.007 at MAP).
**Why:** membership-probability readout is strongly preferred; penalising targets of broad concepts is wrong for these data.
**How to apply:** don't spend slots on target-side size-principle / typicality readouts. MAP screen script: iter_4/candidate_2/scratch/map/run.py (pass only value_vars to compile_fn). Related: [[decision-rule-map-screen]], [[interval-prefix-sum-speedup]].

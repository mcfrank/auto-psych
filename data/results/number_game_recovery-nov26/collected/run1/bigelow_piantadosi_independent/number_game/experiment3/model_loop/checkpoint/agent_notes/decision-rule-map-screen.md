---
name: decision-rule-map-screen
description: MAP screen of decision rules on top of the incumbent number-game learner (experiment3 iter_3) — which response rules the data support
metadata:
  type: project
---
2026-10-31, experiment3 iter_3/candidate_2 (decision-rule brief). The incumbent gamma_shaped_range_width_prior with a swapped response stage was screened by MAP (find_MAP + observedlogp, ~2 min each, in parallel; script /tmp/drule/run.py):
base -2556.6; log-odds yes-bias -> bias 0.04 (null); lapse+guess -> lapse 7e-7 (null); odds power d -> 0.895, -2554.4 (+2.2); OR-of-2 sampled concepts -2659; majority-of-3 -2635; AND-of-2 -2785.
**Why:** bias and lapse fit to the null, so the novelty gate would reject them (see [[interval-prefix-sum-speedup]] gotcha); sampling-based rules lose badly.
**How to apply:** submitted conservative_belief_reporting (odds^d, d ~ Beta(3,1), so it can only compress). Don't re-spend slots on lapse or bias for this learner. MAP-screen a nested param before committing.

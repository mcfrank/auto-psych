---
name: previous-trial-concept-carryover
description: exp3 iter_4 c2: proactive-interference model (previous trial's set concept mixed in); MAP carry ~0.002 (null); no trial-position drift either; batched_dot rewrite 0.046 s/grad
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_2 (exploratory slot): `previous_trial_concept_carryover` = incumbent sequence_opens_bottom_ends_top with p = (1-carry)*P(current set) + carry*P(previous row's set, same session via trial_index resets). Sets are interleaved (6149 set changes / 6750 rows). MAP LL -3090.6, carry ≈ 0.002: lag-1 sequential effects are null. Raw residuals by trial-position quintile are all |<0.006|, so there's no fatigue or practice drift either.
The first version used a posterior[cell_set] gather with 2130 cells: 0.109 s per gradient, 353 s per smoke run (too slow). The batched_dot blocks of 8 brought it to 0.046 s. Pitfall hit again: `pkill -f "candidate-dir..."` killed my own shell.
**Why:** sequential/trial-order process families are now tested and null. **How to apply:** don't propose carryover, drift or fatigue models. Check OK: smoke LL -3093.8, 90 s per run. Check whether this one failed novelty (likely, since carry ≈ 0). Related: [[session-heterogeneity]], [[pymc-model-speed-patterns]], [[top-marking-uncertain-sampling]]

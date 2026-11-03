---
name: exp3-iter0-drifting-criterion
description: Exp3 decision-rule slot — log-odds yes criterion drifting with trial_index ties base (-3194.0 vs -3193.9 smoke); stimuli explain session drift
metadata:
  type: project
---
Exp3 iter 0 candidate_0 (2026-10-31), `drifting_yes_criterion`: digit_sum_open_ended_range_discount inference plus p = sigmoid(logit(p_concept) + drift*(trial_index/450-0.5)), with a lapse. Smoke LL -3194.0, base -3193.9. So the raw yes-rate drop (0.51 to 0.35) comes from stimulus order, not from criterion drift. Submitted anyway. trial_index does reach prepare_observed (the task's raw_columns include it); the novelty pool lacks it, so I default it to session start.

**Why:** like [[soft-criterion-decision-rule]] and [[exp2-iter1-majority-vote]], decision-rule changes tie.
**How to apply:** don't propose session-drift/fatigue models on exp3. A possible novelty-duplicate risk if drift≈0.

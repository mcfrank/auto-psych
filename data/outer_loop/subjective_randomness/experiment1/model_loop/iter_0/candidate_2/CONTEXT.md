# Inner Loop — round 0, candidate 2 of 3

Responses CSV: `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/responses.csv`
Feature columns available (use these as `pm.Data` names): `participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model,n_a,h_a,alts_a,max_run_a,n_b,h_b,alts_b,max_run_b,p_a,p_alts_a,max_run_norm_a,imbalance_a,periodicity_a,p_b,p_alts_b,max_run_norm_b,imbalance_b,periodicity_b`

Work in two steps:
1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.
2. Write `candidate.py` — a module-level PyMC model implementing only that
   hypothesis.

`existing_hypotheses.md` lists the hypotheses already in the model set and
how well each fits. Read it so you propose a *distinct* or *refined*
hypothesis — never a blend of several.

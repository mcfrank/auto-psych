# Inner Loop — round 0, candidate 0 of 3

Responses CSV: `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/responses.csv`
Columns in the responses CSV: `sequence_a,sequence_b,n_a,h_a,alts_a,max_run_a,rep_motifs_a,alt_motifs_a,sym1_a,sym2_a,sym3_a,sym4_a,sym5_a,sym6_a,sym7_a,sym8_a,n_b,h_b,alts_b,max_run_b,rep_motifs_b,alt_motifs_b,sym1_b,sym2_b,sym3_b,sym4_b,sym5_b,sym6_b,sym7_b,sym8_b,p_a,p_alts_a,max_run_norm_a,imbalance_a,periodicity_a,occ_n10_a,occ_n20_a,occ_n50_a,local_imbalance_a,p_b,p_alts_b,max_run_norm_b,imbalance_b,periodicity_b,occ_n10_b,occ_n20_b,occ_n50_b,local_imbalance_b,participant_id,trial_index,chose_left,generating_model`

Read the columns you need as `pm.Data` containers, matching each container name to a column. **Only numeric columns can back a `pm.Data`** — the precomputed integer/float feature columns and `chose_left`.

The raw H/T sequence strings `sequence_a` and `sequence_b` are **not numeric** and cannot be a `pm.Data` directly. To make your hypothesis depend on an aspect of the sequence the precomputed features discard — order, position, recency, or specific sub-sequences — define a module-level `compute_features(sequence_a, sequence_b) -> dict[str, float]` in `candidate.py`. The pipeline runs it on the raw sequences for every trial and exposes each returned key as a column you read with a matching `pm.Data`. This extends the feature space beyond the precomputed columns above.

Work in three steps:
1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.
2. Write `model_name.txt` — a short snake_case name for the model (it
   becomes the model's identifier everywhere downstream).
3. Write `candidate.py` — a module-level PyMC model implementing only that
   hypothesis.

`existing_hypotheses.md` lists the hypotheses already in the model set and
how well each fits. Read it so you propose a *distinct* or *refined*
hypothesis — never a blend of several — under a name not already taken.

`critiques.md` (/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/iter_0/critique/critiques.md) is a posterior-predictive critique of
the current **best** model: the test statistics on which it significantly
fails to reproduce the data, each with the direction of the discrepancy
and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not
confirmatory tests — several are checked per round, so prefer a
discrepancy that survives the FDR (`q ≤ alpha`). Use the strongest such
discrepancy to motivate a single mechanism that would close that gap.

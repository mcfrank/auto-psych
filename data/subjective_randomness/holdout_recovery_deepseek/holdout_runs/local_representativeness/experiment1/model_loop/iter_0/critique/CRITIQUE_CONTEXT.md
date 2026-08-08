# Critique context

**Incumbent (best) model:** `minkowski_accumulated_typicality`
**Incumbent model code:** `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/models/minkowski_accumulated_typicality.py`
**Incumbent hypothesis:** People evaluate the randomness of a sequence by accumulating a subjective
sense of typicality over its length, computing each event's typicality as
a penalty based on its distance from a mental prototype, with the distance
raised to a freely inferred Minkowski-like exponent so extreme feature
deviations are disproportionately punished.

**Responses CSV:** `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `sequence_a,sequence_b,n_a,h_a,alts_a,max_run_a,rep_motifs_a,alt_motifs_a,sym1_a,sym2_a,sym3_a,sym4_a,sym5_a,sym6_a,sym7_a,sym8_a,n_b,h_b,alts_b,max_run_b,rep_motifs_b,alt_motifs_b,sym1_b,sym2_b,sym3_b,sym4_b,sym5_b,sym6_b,sym7_b,sym8_b,p_a,p_alts_a,max_run_norm_a,imbalance_a,periodicity_a,occ_n10_a,occ_n20_a,occ_n50_a,local_imbalance_a,p_b,p_alts_b,max_run_norm_b,imbalance_b,periodicity_b,occ_n10_b,occ_n20_b,occ_n50_b,local_imbalance_b,participant_id,trial_index,chose_left,generating_model`
**Model set directory:** `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/models`

Propose **8** test statistics. Write each to `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/iter_0/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/iter_0/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/responses.csv \
    --model minkowski_accumulated_typicality \
    --models-dir /Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/models \
    --test-stats-dir /Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/iter_0/critique/test_stats \
    --out /Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/iter_0/critique/ppc_results.json \
    --cache-dir /Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_faithful/mcmc_cache \
    --n-replicates 200 \
    --significance-alpha 0.05
```

That writes `/Users/ben/Documents/auto-psych/data/subjective_randomness/holdout_recovery_deepseek/holdout_runs/local_representativeness/experiment1/model_loop/iter_0/critique/ppc_results.json` with a two-sided empirical p-value per statistic (200 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

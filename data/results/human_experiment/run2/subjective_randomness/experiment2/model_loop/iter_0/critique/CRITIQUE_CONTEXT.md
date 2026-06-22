# Critique context

**Incumbent (best) model:** `inner_loop_model`
**Incumbent model code:** `/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models/inner_loop_model.py`
**Incumbent hypothesis:** Random-looking sequences are judged by their Euclidean distance to a messy prototype with a strictly positive ideal imbalance and an ideal alternation rate, with quadratic asymmetric penalties.

**Responses CSV:** `/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model,n_a,h_a,alts_a,max_run_a,rep_motifs_a,alt_motifs_a,n_b,h_b,alts_b,max_run_b,rep_motifs_b,alt_motifs_b,p_a,p_alts_a,max_run_norm_a,imbalance_a,periodicity_a,p_b,p_alts_b,max_run_norm_b,imbalance_b,periodicity_b`
**Model set directory:** `/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/iter_0/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/iter_0/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/responses.csv \
    --model inner_loop_model \
    --models-dir /scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/models \
    --test-stats-dir /scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/iter_0/critique/test_stats \
    --out /scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/iter_0/critique/ppc_results.json \
    --cache-dir /scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/iter_0/critique/.fit_cache \
    --n-replicates 200 \
    --significance-alpha 0.05
```

That writes `/scratch/users/benpry/auto-psych/outer_loop_live/run2/data/subjective_randomness/experiment2/model_loop/iter_0/critique/ppc_results.json` with a two-sided empirical p-value per statistic (200 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

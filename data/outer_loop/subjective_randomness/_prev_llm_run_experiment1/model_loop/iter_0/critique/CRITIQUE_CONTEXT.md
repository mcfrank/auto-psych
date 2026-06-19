# Critique context

**Incumbent (best) model:** `bayesian_diagnosticity`
**Incumbent model code:** `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/models/bayesian_diagnosticity.py`
**Incumbent hypothesis:** Random-looking sequences are diagnostic of a fair coin over salient
non-random alternatives: alternating, biased, and streaky generators.

**Responses CSV:** `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `participant_id,trial_index,sequence_a,sequence_b,chose_left,chose_right,model,n_a,h_a,alts_a,max_run_a,rep_motifs_a,alt_motifs_a,n_b,h_b,alts_b,max_run_b,rep_motifs_b,alt_motifs_b,p_a,p_alts_a,max_run_norm_a,imbalance_a,periodicity_a,p_b,p_alts_b,max_run_norm_b,imbalance_b,periodicity_b`
**Model set directory:** `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/models`

Propose **8** test statistics. Write each to `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/iter_0/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/iter_0/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/responses.csv \
    --model bayesian_diagnosticity \
    --models-dir /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/models \
    --test-stats-dir /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/iter_0/critique/test_stats \
    --out /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/iter_0/critique/ppc_results.json \
    --cache-dir /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/iter_0/critique/.fit_cache \
    --n-replicates 200 \
    --significance-alpha 0.05
```

That writes `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment1/model_loop/iter_0/critique/ppc_results.json` with a two-sided empirical p-value and a Benjamini–Hochberg FDR-adjusted p-value per statistic (200 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value_adjusted` ≤ 0.05.

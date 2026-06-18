# Critique context

**Incumbent (best) model:** `bayesian_fair_coin`
**Incumbent model code:** `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/models/bayesian_fair_coin.py`
**Incumbent hypothesis:** Observers compare two binary sequences via the log Bayes factor between a fair-coin null and a biased-coin alternative.

**Responses CSV:** `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model,n_a,h_a,alts_a,max_run_a,n_b,h_b,alts_b,max_run_b,p_a,p_alts_a,max_run_norm_a,imbalance_a,periodicity_a,p_b,p_alts_b,max_run_norm_b,imbalance_b,periodicity_b`
**Model set directory:** `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/iter_1/critique/test_stats/<name>.py`.

Run the posterior-predictive harness once over that directory:

```bash
python3 -m src.critique.ppc \
    --responses /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/responses.csv \
    --model bayesian_fair_coin \
    --models-dir /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/models \
    --test-stats-dir /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/iter_1/critique/test_stats \
    --out /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/iter_1/critique/ppc_results.json \
    --cache-dir /Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/iter_1/critique/.fit_cache \
    --n-replicates 200 \
    --significance-alpha 0.05
```

It writes `/Users/ben/Documents/auto-psych/data/outer_loop/subjective_randomness/experiment2/model_loop/iter_1/critique/ppc_results.json` with a two-sided empirical p-value and a Benjamini–Hochberg FDR-adjusted p-value per statistic (200 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value_adjusted` ≤ 0.05.

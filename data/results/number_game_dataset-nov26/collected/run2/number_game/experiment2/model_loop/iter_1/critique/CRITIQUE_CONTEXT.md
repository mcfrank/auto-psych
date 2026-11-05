# Critique context

# The task

Each participant sees a few example numbers between 1 and 100, produced by a
hidden computer program, and is asked about other numbers between 1 and 100:
would the program also produce this number? The examples on a trial are its
*set* (one to four numbers, shown in the order given); the number asked about
is the *target*. The participant answers yes or no. There is no correct
answer and no feedback.

Every participant answers about many (set, target) pairs, and the same set is
asked about with many targets.

## The data

Each row of `responses.csv` is one answer:

| Column           | Meaning                                                            |
| ---------------- | ------------------------------------------------------------------ |
| `number_set`     | the example numbers, comma-separated in the order shown ("16,8,64") |
| `target`         | the number asked about, 1-100                                      |
| `rating`         | 1 if the participant answered yes (the program would produce it), else 0 |
| `trial_index`    | the trial's position in that participant's session                 |

## Your job

**Incumbent (best) model:** `seed_overshadowing_last_digit_recency`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/seed_overshadowing_last_digit_recency.py`
**Incumbent hypothesis:** Refinement of `seed_multiple_digit_overshadowing_overlap` by one grafted component, taken from `last_example_digit_recency_overlap`: people read the examples in order, and the final digit of the last example shown lingers in their guess as a candidate pattern ("ends in 1") even when the earlier examples do not share it. Otherwise the judgment is unchanged: "yes" follows the salience-weighted share of the examples' shared profile the target has, with "multiples of the smallest example" in that profile when it divides all the others, and a final digit visibly shared by all examples overshadowing every other property. Compared with the seed-overshadowing model, targets that share the last-shown example's final digit are endorsed more than targets that share the first-shown example's, so the same examples in a different order are generalized differently.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/iter_1/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/iter_1/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/responses.csv \
    --model seed_overshadowing_last_digit_recency \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/iter_1/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/iter_1/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/iter_1/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

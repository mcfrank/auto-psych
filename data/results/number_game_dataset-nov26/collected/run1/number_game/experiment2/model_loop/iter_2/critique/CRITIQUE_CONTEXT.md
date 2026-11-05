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

**Incumbent (best) model:** `multiple_route_exemplar`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/multiple_route_exemplar.py`
**Incumbent hypothesis:** People judge a target by its summed similarity to the examples (closeness in magnitude, same final digit, same parity, and whether it divides by the set's common divisor), sharpened when the examples share a suspicious coincidence. On top of that there is a second, separate way for a target to resemble an example: being one of its multiples (14, 49 or 98 after "7"; 64 after "4"). This multiples route adds a fixed amount of similarity no matter how far apart the two numbers are, instead of only scaling down closeness. So multiples of an example that are far from it and share none of its digits are still often endorsed, especially after single examples and small-number examples, where magnitude similarity alone predicts almost certain rejection.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/iter_2/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/iter_2/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/responses.csv \
    --model multiple_route_exemplar \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/iter_2/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/iter_2/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/iter_2/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

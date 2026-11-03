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

**Incumbent (best) model:** `extended_multiples_concepts`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/extended_multiples_concepts.py`
**Incumbent hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples drawn from the concept without replacement), but their space of arithmetic concepts is larger: besides the familiar multiples of 3-12, they consider "multiples of m" for any m from 13 to 50 (multiples of 15, 20, 25, 45, ...), each such less-familiar multiples rule equally likely a priori and the family as a whole taking an inferred share of the rule prior. Because these rules are small, the size principle makes them dominate when every example is a multiple of a larger number (e.g. 45, 90 or 25, 75, 100), predicting fewer "yes" answers to targets that are only multiples of the smaller familiar factor (e.g. 27 or 60 after 90, 45).

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/responses.csv \
    --model extended_multiples_concepts \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/iter_3/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

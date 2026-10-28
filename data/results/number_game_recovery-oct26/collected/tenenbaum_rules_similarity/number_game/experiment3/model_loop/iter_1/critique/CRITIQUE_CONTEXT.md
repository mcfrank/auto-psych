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

**Incumbent (best) model:** `last_example_marks_range_top`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models/last_example_marks_range_top.py`
**Incumbent hypothesis:** Refinement of `top_marking_range_sampling`: people do the same Bayesian concept learning over arithmetic rules (primes excluding 1), every range of 1-100 and range-restricted rules, and believe that, with some inferred odds, the program deliberately shows its range's top number and draws the other examples at random — but they believe the program ends its demonstration on that top number. The one change: the marked top is read from the order of presentation, so a range ending at the largest example gets the generative top-marking credit (larger for wider ranges) only when the largest example is the last one shown, and then at full strength whatever the set size, while sets whose largest example comes earlier get no edge credit — after 20,35,60 people say "yes" to 45 but not 62, whereas after 60,35,20 they extend above 60 as readily as below 20.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/responses.csv \
    --model last_example_marks_range_top \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/45bd45c333332e1a/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

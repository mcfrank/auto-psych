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

**Incumbent (best) model:** `fives_privileged_decoupled_prime_odd`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models/fives_privileged_decoupled_prime_odd.py`
**Incumbent hypothesis:** Refinement of `prime_bonus_odd_cost_decoupled`, with everything in it kept: primitive-built rules with a per-primitive cost, the final-digit discount, costs on powers/cubes and squares, times-table-graded multiples, a separate fitted bonus for "prime numbers" and cost for "odd numbers", the repeated-digit rule, Erlang ranges with a bonus for whole hundred-chart rows, the size principle and near-all-or-none commitment. The one change is that "multiples of 5" is unlike the other times tables. Counting by fives is a drilled skill, and membership can be read straight off the numeral (it ends in 0 or 5), so this one rule gets its own fitted prior bonus. So after sets of multiples of 5 that are not all multiples of 10 (e.g. 90, 100, 5 or 35, 15, 85), people commit to "multiples of 5": they say yes to other numbers ending in 0 or 5 and no to neighbouring non-multiples more sharply than the base model predicts, while sets that fit other times tables are unaffected.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/responses.csv \
    --model fives_privileged_decoupled_prime_odd \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

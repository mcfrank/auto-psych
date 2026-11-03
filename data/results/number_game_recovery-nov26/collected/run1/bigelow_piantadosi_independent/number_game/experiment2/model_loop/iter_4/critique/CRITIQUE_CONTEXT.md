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

**Incumbent (best) model:** `rule_plus_exception_neighbourhood`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models/rule_plus_exception_neighbourhood.py`
**Incumbent hypothesis:** Refinement of `rule_plus_exception_concepts`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus exception" concepts, but the exception they add to a rule is not a single isolated number — it is a small fuzzy region of the number line around it ("ends in 2, and also numbers around 27"), whose radius (zero, one, two … up to three steps) is a priori more likely the smaller it is, with the fall-off fitted. The one change is this neighbourhood-sized exception: when one example breaks the others' pattern (for instance, the one odd number among even ones), people still endorse it and still extend the rule to distant numbers, but they also endorse the off-pattern example's close neighbours more than the point-exception model predicts, as the critique reports.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/responses.csv \
    --model rule_plus_exception_neighbourhood \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

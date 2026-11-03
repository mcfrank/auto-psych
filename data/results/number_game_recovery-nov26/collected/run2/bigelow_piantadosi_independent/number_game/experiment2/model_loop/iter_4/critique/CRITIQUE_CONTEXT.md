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

**Incumbent (best) model:** `rule_familiarity_within_kinds`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/rule_familiarity_within_kinds.py`
**Incumbent hypothesis:** Refinement of `base_ten_multiples_capacity_limited`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with a prior plausibility for each kind of rule (odd, even, primes, squares/cubes, powers, multiples, final digit, contained digit, and multiples of 5 as its own base-ten kind), but the rules within a kind are not equally familiar — each rule carries its own familiarity that raises or lowers its prior plausibility relative to the other rules of its kind (a substantial, fixed amount of variation, with each rule's familiarity inferred from the data). The single change is replacing the equal sharing of a kind's plausibility among its rules with these rule-specific familiarities, predicting that some rules (e.g. multiples of 7, or "contains the digit 1") are rarely entertained even when the examples fit them, so those sets are generalized by range rather than by the rule, while other rules of the same kind (e.g. powers of 10, multiples of 11) are adopted readily.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/responses.csv \
    --model rule_familiarity_within_kinds \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/iter_4/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

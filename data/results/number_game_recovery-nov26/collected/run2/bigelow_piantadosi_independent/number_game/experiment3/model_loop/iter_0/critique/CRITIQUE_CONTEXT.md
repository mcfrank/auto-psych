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

**Incumbent (best) model:** `parity_veto_sparse_familiar_rules`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/parity_veto_sparse_familiar_rules.py`
**Incumbent hypothesis:** Refinement of `parity_conflict_decision_veto`: people infer the hidden concept by capacity-limited Bayesian learning over rules (multiples of 5 as its own base-ten kind) and numerical ranges, and a target whose parity matches none of the examples triggers an automatic parity conflict that, on a fixed fraction of such trials, overrides the inferred concept with a "no". The single change is grafting sparse rule familiarity (from `sparse_rule_familiarity_within_kinds`) onto the concept-learning stage: most rules share their kind's plausibility, and only a few stand out as exceptionally familiar (multiples of 11, powers of 10) or unfamiliar (multiples of 7, "contains the digit 1"). The claim is that the decision-stage parity veto survives once rule priors are calibrated this way: opposite-parity targets in all-odd or all-even sets (e.g. odd numbers inside the span of 20, 32, 92) get fewer yeses than the inferred concept implies, while same-parity targets are unaffected.

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/iter_0/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/iter_0/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/responses.csv \
    --model parity_veto_sparse_familiar_rules \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/iter_0/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/iter_0/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/iter_0/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

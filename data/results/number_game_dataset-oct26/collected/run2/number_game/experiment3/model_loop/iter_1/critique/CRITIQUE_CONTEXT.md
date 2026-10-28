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

**Incumbent (best) model:** `enumerated_list_concept_exemplar`
**Incumbent model code:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/enumerated_list_concept_exemplar.py`
**Incumbent hypothesis:** Refinement of zigzag_breadth_sequence_exemplar (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g" and "ends in d" rules, the relaxing criterion, the filled-in span, the momentum of the sequence's direction of travel and the zigzag-set breadth). The single change adds one concept to the space people consider: that the program simply produces a fixed list of particular numbers, of which the shown examples are members. This concept is consulted as separate evidence at the decision, and it applies only to targets that are themselves among the shown examples. People therefore say "yes" to a repeated example far more often than to a number only 1-3 away, even though the two are almost equally similar in size (the critique's exact-example boost).

**Responses CSV:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/responses.csv`
**Columns (DataFrame your test statistics receive):** `number_set,target,trial_index,rating`
**Model set directory:** `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models`

Propose **8** test statistics. Write each to `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats/<name>.py` as a function `test_statistic(df)` returning a scalar, with `# name:` and `# description:` header comments.

You do **not** need to run anything. After you write the statistics, the pipeline runs the posterior-predictive harness automatically over `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats` and records the results:

```bash
python3 -m src.critique.ppc \
    --responses /scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/responses.csv \
    --model enumerated_list_concept_exemplar \
    --models-dir /scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models \
    --test-stats-dir /scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/test_stats \
    --out /scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/ppc_results.json \
    --cache-dir /scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/mcmc_cache \
    --n-replicates 1000 \
    --significance-alpha 0.05
```

That writes `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/iter_1/critique/ppc_results.json` with a two-sided empirical p-value per statistic (1000 posterior-predictive replicates). A statistic is a **significant discrepancy** when its `p_value` ≤ 0.05 (raw, no multiple-comparisons correction).

Each statistic is called 1001 times (the observed data, then every replicate). Each call must finish within 5 s and all of them within 300 s (about 0.3 s per call), so vectorise it: no row-wise `apply` and no Python loops over rows. A statistic that raises or runs out of time gets no p-value.

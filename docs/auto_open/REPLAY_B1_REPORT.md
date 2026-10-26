# Run B′ (replay of run B's experiment 1 on the fixed code): result

From the local Sherlock session, 2026-10-26, for the driver session. Handoff:
`HANDOFF_rehearsal.md`, "Run B′". Comparison: `RUN_B_REPORT.md`.

**Run B′ completed in 19 h 19 min, for $38.46.** On the same design, data, folds and starting
theories as run B's experiment 1, the fixed code:
- **admitted 13 of 25 attempts (52%),** against run B's 9 of 38 (24%);
- **cut too-slow rejections to 8,** against 24;
- **ended with an incumbent 100 nats better on experiment 1's held-out data,** with no stalled or
  failed fold.

The cost:
- **The live set narrowed to one family** (`contextual_contrast_*` / `type_contrast_*`).
- **A starting theory was pruned** that is still about 110 nats better on E than the new incumbent
  (§4).

`gt_name_mentions.txt` is empty.

## 1. Job and bring-back

| job | code | work root | outcome |
|---|---|---|---|
| **49502878_0** | **fc632d4b** | `rsa_open_replay_b1` (fresh) | **COMPLETED, Sun 08:55 → Mon 04:14 (19 h 19 min), 24 cores** |

- **The replay was set up as intended.** The log shows `replay: experiment 1's design and data
  from …/e6344f63e5fb0046/…/chain_0`, and no designer ran.
- **The ground truth is run B's.** `theory_sha256` is the same as in `rehearsal_b/ground_truth.json`.

Brought back to `data/rsa_open/replay_b1/` as for run B:
- Included: `outputs/`, with the `.fit_cache/*.json` sidecars force-added.
- Included: `ground_truth.json`, `gt_name_mentions.txt`, `DONE`, `code_commit` and the job log as
  `job_49502878_log.txt`.
- Left out: the fit caches, agent scratch, `ground_truth.py` and every copy of E's trials.
- There is no `code_history`: a fresh root is staged once.

## 2. Admission limit

```
[open loop] admission limit 59 min: 2 x the median of the starting fits (competitor_confusion_speaker 26,
isolated_graded_costly_l3 160, oddity_heuristic_listener 13, rsa_l1 16, rsa_l2 33,
rsa_l2_singleton_feat_color_valence_l0 55 min)
```

The limit is 59 min, not the handoff's estimate of about 90. The median of the six starting fits
is 29.5 min, and 2 × 29.5 = 59. It is recorded in `history.json` as `admission_limit_sec` 3543.6.

## 3. Rounds, against run B's experiment 1

| round | B′ attempts | admitted | too slow | not novel | B attempts | admitted | too slow | other |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | **4** | 3 | 1 | 11 | 1 | 10 | — |
| 2 | 9 | **3** | 4 | 2 | 8 | 4 | 3 | 1 no file |
| 3 | 8 | **6** | 1 | 1 | 8 | 3 | 3 | 1 no file, 1 not novel |
| 4 | — | — | — | — | 11 | 1 | 8 | 2 not novel |
| **total** | **25** | **13** | **8** | 4 | 38 | 9 | 24 | 5 |

- **Not converged:** no rejection for convergence, and **no `cv_problem`** in any step of
  `history.json`.
- **The 3× fold bound and the near-miss rule never fired.**
- **Rounds:** B′ was capped at 3 rounds (`MAX_ITERATIONS=3`). Run B ran 4 of 5 before its stale
  stop.
- **Round 3 was the most productive:** 6 admitted, 1 too slow. Two of the six came from a retry or
  a repair.

### Incumbent per step

| step | B′ | B |
|---|---|---|
| start | rsa_l2_singleton_feat_color_valence_l0 | the same |
| round 1 | **contextual_contrast_listener** (discovered, explore) | rsa_l2_singleton_feat_color_valence_l0 |
| round 2 | contextual_contrast_listener | **bounded_depth_listener** |
| round 3 | contextual_contrast_listener (refinements admitted) | bounded_depth_listener |

### The sampling lines

- **The theorists' self-check:** "Projected admission fit on all the data" appears in 26 of the
  30 agent transcripts. It comes with the step size and correlated pairs, for example "alpha and
  cost_weight are correlated at -0.89".
- **Rejections carry the report only when warmup ended before the limit.** Two of the 8 did, for
  example: "warmup took 41 minutes and it was killed while sampling. NUTS took 16.3 leapfrog steps
  per draw by the end of warmup, at step size 0.369: many (above about 10), so the posterior is
  narrow or curved…".
- **The other 6 say only** "it was still in warmup, adapting its step size: the posterior is hard
  to explore". A fit killed in warmup has no settled step count to report.

## 4. Final standing and the comparison with B

Held-out ELPD (grouped CV), the same folds in both runs. The start step's numbers are identical in
B and B′.

| | incumbent | exp1 elpd_cv | E elpd_cv |
|---|---|---|---|
| start (both) | rsa_l2_singleton_feat_color_valence_l0 | −7082.6 | **−20647.9** |
| run B, end | bounded_depth_listener | −7040.3 | −20843.0 |
| **run B′, end** | **contextual_contrast_listener** | **−6940.3** | −20758.2 |

- **On exp1, B′'s incumbent is 100 nats ahead of B's,** and 142 ahead of the start.
- **On E, B′'s incumbent is 110 nats behind the starting incumbent.** B's was 195 behind.
- These are point differences between models' summed held-out LPDs. I did not compute a paired SE
  across the two runs.

B′ ends with 5 live models. "Behind" is the nats behind the best of these 5 ± the clustered SE:

| theory | E | exp1 |
|---|---|---|
| **contextual_contrast_listener** | 15.0 ± 9.3 | **best** |
| contextual_contrast_isolated_listener | 15.8 ± 6.5 | 1.1 ± 2.7 |
| contextual_contrast_distinct_listener | **best** | 1.9 ± 5.9 |
| type_contrast_distinct_listener | 13.4 ± 11.5 | 27.0 ± 15.6 |
| type_contrast_listener | 31.8 ± 16.9 | 27.3 ± 15.6 |

- **One family:** all five are `contextual_contrast_*` or `type_contrast_*`, and three of them
  refine the incumbent.
- **Every starting theory was pruned at the end.**
  - Three were more than 4 SEs behind the best on E: `isolated_graded_costly_l3`,
    `oddity_heuristic_listener` and `rsa_l1`.
  - Three were beyond 2 × the clustered SE on exp1: `competitor_confusion_speaker`, `rsa_l2` and
    `rsa_l2_singleton_feat_color_valence_l0`.
  - `rsa_l2_singleton_feat_color_valence_l0` was 142.3 nats behind the incumbent on exp1 (2 ×
    the clustered SE is 84.6), though it is about 95–110 nats better on E than every carried model.
    Run B's experiment 1 kept it.
- **Taken together:** the next experiment's design would compare five variants of one mechanism,
  and the literature's best model would be gone from the carried set. The pruning rule judges on
  the newest block only. Whether E should weigh in, beyond the 4-SE eligibility guard, is a
  question for you.

## 5. Critic

| round | significant after FDR |
|---|---|
| 1 | 8 / 8 |
| 2 | **2 / 8** (3 nominally) |
| 3 | 8 / 8 |

Round 2 was the first critique in runs A, B and B′ that was not 8 of 8. In round 3 the critic's
first attempt wrote no usable statistic ("attempt 1: no usable test statistic"); the retry gave 8
of 8.

## 6. Spend (`outputs/token_usage_summary.json`)

| source | calls | cost |
|---|---|---|
| theorists | 26 | $31.09 |
| critic | 4 | $7.37 |
| **total** | | **$38.46** (estimate $30–40) |

There was no designer and no pilot: the design was replayed.

## 7. Timeline

| time | stage |
|---|---|
| 08:55–09:13 | staging and the agent tree |
| 09:13–13:47 | starting fits and their folds, 4.6 h |
| 13:47 → Mon 04:12 | 3 rounds, about 4.8 h each |

## 8. Ground-truth check (`src/rsa_open/oracle.py`, job 49631393, 2026-10-26)

The check took 2 h 23 min, with no agents and no API spend.
- **What it does:** it fits the hidden truth's theory on run B′'s own corpus and folds, beside the
  19 theories the loop ended with or pruned. Those come from the loop's cache. Rejected candidates
  are not scored.
- **Where the results are:** `data/rsa_open/replay_b1/oracle/{oracle.json,oracle.md}` and the job
  log. They name the truth only as `ground_truth`.
- **How to read it:** "− truth" is a theory's ELPD-CV minus the truth's, ± the clustered SE.
  exp1 was generated by the truth; E is real people.

| theory | status | exp1 − truth | E − truth |
|---|---|---|---|
| contextual_contrast_listener (incumbent) | live | **−11.4 ± 6.2** | −18.0 ± 18.7 |
| contextual_contrast_isolated_listener | live | −12.5 ± 5.3 | −18.8 ± 16.3 |
| contextual_contrast_distinct_listener | live | −13.3 ± 4.9 | −3.0 ± 13.6 |
| type_contrast_distinct_listener | live | −38.3 ± 14.1 | −16.4 ± 19.0 |
| type_contrast_listener | live | −38.6 ± 16.1 | −34.8 ± 24.8 |
| aspect_qud_listener (best pruned) | pruned | −68.3 ± 15.5 | −26.0 ± 27.9 |
| rsa_l2_singleton_feat_color_valence_l0 (the starting incumbent) | pruned | −153.7 ± 41.8 | **+92.2 ± 60.6** |
| rsa_l2_uncompounded_l1 | pruned | −154.1 ± 33.1 | +65.7 ± 66.3 |
| the other 11 pruned | pruned | −139.7 to −1408.3 | −106.8 to −1680.8 |

On exp1, the truth's own data:
- **The loop closed most of the gap to the truth.** The starting incumbent was 153.7 ± 41.8 nats
  behind the truth; the final incumbent is 11.4 ± 6.2 behind. That is about 93% of the distance.
- **The truth is still ahead.** The incumbent trails it by 1.8 SE, and the other two
  `contextual_contrast_*` models by 2.4–2.7 SE.
- **Run B's incumbent** `bounded_depth_listener` (exp1 −7040.3) would be about 111 nats behind the
  truth (−6928.9). This is on the same data and folds, but it's a point difference only, since run
  B's model was not in this check.

On E:
- **The truth does not lead.** The pruned starting incumbent beats it by 92.2 ± 60.6 (1.5 SE).
- **The carried models are within 1.4 SE of the truth on E.**
- So the pruning in §4 removed the model that predicts real people best, but the truth itself is
  not better on E than what was carried. On this evidence, losing it does not hurt recovery of the
  truth.

# Run B (the rehearsal): status after one day, and one blocking fit

## Update, Sat 2026-10-24 09:16: experiments 1 and 2 done, experiment 3 in design

**Job 49021326_0 is still running:** 47 h 27 min elapsed, 3 days 0 h 33 min left (the limit is
Tue 09:49). The stall in §3 below resolved itself, but it cost 9.4 h for nothing:
- The refit of `feature_parsimony_listener`'s 5th fold finished at Fri 12:35, after 5 h 14 min.
- It still failed the gate (max R-hat 1.528, min bulk ESS 7).
- The model was therefore scored `cv_converged: false`, `trusted: false`, `eligible: false`.

Round 3's wait (03:12 → 12:35) bought nothing. That is the case §8 options 1 and 2 would have cut
short. No redeploy is needed for this run now; the choice matters for the next campaign.

### Per experiment

| | experiment 1 | experiment 2 | experiment 3 |
|---|---|---|---|
| design fits | 15 design fits, none screened out | 9 from the previous loop + 9 design fits, none screened out | 5 from the previous loop + 12 design fits (1 not converged: min bulk ESS 77); **2 screened out at the 2 h limit**: `familiar_isolated_graded_costly_l2`, `softmax_surprisal_isolated_costly_l4` |
| paradigm | `graded_belief_allocation`: 10-point bet | `probed_referent_likelihood_rating`: rating | designer running since Sat 08:19 (cycle 1) |
| size | N = 120 × 15 trials, 3.9 min | N = 200 × 20 trials, 3.0 min | |
| EIG (bits/$, floor) | 3.50 bits (0.0337, floor 0.0120) | 3.87 bits (0.0291, floor 0.0129) | |
| pilot (comprehension, catch, readable) | 1.0, 1.0, 1.0; 4.2 min estimated | 1.0, 1.0, 1.0; 3.1 min estimated | |
| design cycles | 1 (+ `design.abandoned_1` from the redeploy) | 1 | |
| loop | 4 of 5 rounds; stopped early (incumbent unchanged 2 rounds) | 2 of 5 rounds; stopped early (same rule) | |
| times | frozen Thu 12:09; loop done Fri 15:29 | frozen Fri 19:07; loop done Sat 06:15 | design fits Sat 06:15–08:15 |
| incumbent | `rsa_l2_singleton_feat_color_valence_l0` → **`bounded_depth_listener`** (discovered, round 2) | **`rsa_l1_singleton_feat_color_valence_l0`** (discovered in experiment 1) | |
| carried | 4: `bounded_depth_feature_parsimony`, `bounded_depth_listener`, `rsa_l1_singleton_feat_color_valence_l0`, `rsa_l2_singleton_feat_color_valence_l0` | 5: the same 4 + `feature_uncertainty_listener_2` | |

Experiment 2's final standing ("behind" is in nats ± the clustered SE; grouped CV, except E):

| theory | E | exp1 | exp2 |
|---|---|---|---|
| rsa_l1_singleton_feat_color_valence_l0 | 45.0 ± 23.4 | **best** | **best** |
| bounded_depth_listener | 166.9 ± 57.9 | 8.6 ± 9.8 | 0.1 ± 20.3 |
| bounded_depth_feature_parsimony | 160.1 ± 57.3 | 10.0 ± 10.2 | 9.8 ± 23.9 |
| rsa_l2_singleton_feat_color_valence_l0 | **best** | 29.4 ± 21.3 | 9.1 ± 62.5 |
| feature_uncertainty_listener_2 | 143.2 ± 56.6 | 36.9 ± 20.0 | 41.0 ± 46.9 |

All five are eligible and trusted. On exp2, everything but `feature_uncertainty_listener_2` is
within 0.5 SE of the incumbent: experiment 2's data barely separate them.

### Candidates (the 30-minute admission limit on all of E)

Experiment 2's ledger continues experiment 1's, so its counts below are its own new lines only.

| | rounds | attempts | admitted | too slow | other |
|---|---|---|---|---|---|
| experiment 1 | 4 | 38 | 9 | **24** | 2 no file, 3 not novel |
| experiment 2 | 2 | 23 | 2 | **19** | 2 hypotheses copied word for word from the refined theory |

- **Too slow: 43 of 61 attempts.**
- Experiment 2's round 1 admitted nothing.
- Experiment 1's round 4 had 8 of 11 too slow.
- The two admissions in experiment 2 came from a retry (`topic_discriminability_listener`) and a
  repair (`feature_uncertainty_listener_2`).

The early stops may be partly a consequence: a round that admits almost nothing can hardly change
the incumbent.

### Critic, truth checks, spend

- **The critic** ran in every round: **8 of 8 statistics significant after FDR, every round** (6 of
  6 rounds so far).
- **`gt_name_mentions.txt`** is written at the end of the job; I'll check it then.
- **Spend** (`chain_0/token_usage.jsonl`) is **$94.27**:

  | source | records | cost |
  |---|---|---|
  | theorists | 67 | $78.13 |
  | designer | 2 | $9.87 |
  | critic | 6 | $6.27 |
  | pilot | 666 | not priced (1.9 M tokens recorded) |

### Timeline (my estimate)

- **Experiment 3's design:** 1–6 h from 08:19.
- **Its loop:** 11 h (experiment 2, two rounds) to 27 h (experiment 1, four rounds, with the stall).

**Expected end: between Sunday morning and Monday morning**, inside the limit. When it ends I bring
it back to `data/rsa_open/rehearsal/` per §3 of the handoff and write the full report.

---

## Earlier status (Fri 2026-10-23)

From the local Sherlock session, 2026-10-23, for the driver session. Handoff: `HANDOFF_rehearsal.md`,
"Redeploy run B". All figures are as of **Fri 09:36 PDT**; after that the SSH session dropped.
**Update, Fri 11:46:** the refit is still running, now 4 h 25 min; the round has waited 8.5 h on this one model. Round 3 is not yet scored.

**Run B is running and I am leaving it running.** Nothing here is a failure, but at this pace it
gets close to its time limit.

## 1. The job

| job | code | work root | state |
|---|---|---|---|
| 49001912 | e8b6acb8 | `rsa_open_rehearsal` | cancelled at 1 h 54 min for the redeploy. It had made the truth (all of E) and the 15 design fits; its designer was 13 min in, kept as `design.abandoned_1` |
| **49021326_0** | **99cb1e28** | **same** | **running since Thu 09:49 on 24 cores; 23 h 46 min at 09:36; limit 5 days (Tue 09:49)** |

The redeploy reused the truth and the design fits: the designer started 7 min after the job did.
`design_screened_out.json` lists no theory.

## 2. Progress: experiment 1, round 3 of 5

| time | event |
|---|---|
| Thu 12:09 | design frozen (one cycle) |
| Thu 12:10 | collection (simulated) |
| 12:10–18:33 | starting fits and round 1 (about 6.4 h) |
| 18:33–23:38 | round 2, then scoring |
| 23:53–03:12 | round 3: critic, theorists, admissions, fold fits |
| **03:12 → still at 09:36** | **one model's last fold fit and its refit (§3)** |

`history.json` records the seed step and rounds 1 and 2. **The incumbent changed in round 2**, from
`rsa_l2_singleton_feat_color_valence_l0` to the discovered `bounded_depth_listener`. Round 3 is not
yet scored.

## 3. The blocking fit

`feature_parsimony_listener` was admitted in round 3, whose full fit finished at 01:27. Its 5 CV
folds were then fitted concurrently, each warm-started:
- **4 folds** finished at 03:06–03:07, about 1.7 h each.
- **The 5th** finished at 07:21, about 5 h 50 min, and failed the gate: 31 divergent of 2,000,
  max R-hat 1.546, min bulk ESS 7.
- **`loop_fit` then refits it** (`refit_settings`: target_accept 0.95, a new seed), from the same
  warm start. **That refit has been running alone, on one core, since 07:21:** 2 h 15 min at 09:36,
  while 23 cores sat idle.

There is no limit on either fit. `Orchestrator.live_cfg` gives a live model's fits
`time_limit_sec=math.inf` (`src/rsa_open/loop/orchestrator.py:148`), and `_cv_one` passes that limit
to the folds. The refit is unconditional, as in the closed loop (`src/rsa/loop/fitting.py`
`loop_fit`): a fit this far from converging is refitted too. Main's PyMC loop refits only near
misses (R-hat ≤ 1.2, ESS ≥ 20; `NEAR_MISS_*`), so it would not have refitted this one.

**The stall can recur:** at every rescoring, for every admitted model whose fold sits on a hard
posterior. If the refit also fails, `_cv_one` still returns, and the standing marks the model
`cv_converged: false` (so not trusted). The cost is time, not a crash.

## 4. The 30-minute admission limit on all of E

The handoff asked me to count these. From `attempted_hypotheses.jsonl`:

| round | admitted | too slow (30 min) | other |
|---|---|---|---|
| 1 | 1 | 10 | — |
| 2 | 4 | 3 | 1 wrote no `candidate.py` |
| 3 | 3 | 3 | 1 not novel (RMSE 0.0013), 1 wrote no `candidate.py` |

**16 of 27 attempts were rejected as too slow, including 10 of round 1's 11.** Repairs did not help:
every repair of a too-slow candidate was too slow again. The theorists' self-check now fits a fifth
of E (99cb1e28), so a candidate can pass it and still miss the limit on all of E. That may be part
of it.

## 5. The critic

It ran in every round: 8 statistics, 8 evaluated, **8 significant after FDR, every round.** Run A
had the same result. A critic that finds every statistic significant tells the theorists little
about which discrepancy matters.

## 6. Spend (the new `chain_0/token_usage.jsonl`; 0 records with usage missing)

| source | model | records | cost |
|---|---|---|---|
| theorists | gemini-3.8-flash | 31 | $33.86 |
| designer | gemini-4-argon | 1 | $4.37 |
| critic | gemini-3.8-flash | 3 | $2.96 |
| pilot | gemini-3.8-flash | 300 | $0.00 (tokens recorded; google-genai reports no cost) |
| **total** | | | **$41.19** |

The abandoned job's 13 min of designer time is not in this log. To price the pilot, the summary
would need a per-token rate.

## 7. Timeline estimate (mine)

Rounds run 3.5–6.5 h when nothing stalls. Remaining:
- rounds 3 (once the refit ends), 4 and 5 of experiment 1;
- experiments 2 and 3, each a design plus 5 rounds, about 25–30 h each.

**That puts the end around Monday**, inside the limit, which ends Tue 09:49. Each stall like
§3's adds 6 h or more, and two or three would exhaust the limit. A resubmit resumes, so a timeout
loses the unfinished stage's attempt, not the run.

## 8. Options (yours)

1. **Bound the live fold fits**, for example by the 30-min admission limit or a few hours, or let
   a fold that fails the gate count as unconverged without the refit. Either way the model is
   scored `cv_converged: false` instead of holding the round.
2. **Refit only near misses,** as main's PyMC loop does.
3. **Let it run.** It costs only compute and time; I resubmit if it hits the limit.

Until you say otherwise, it keeps running (option 3). A fix would need a redeploy like the last
one: cancel, pull, `RESTAGE_REASON`, resubmit in the same root. The unfinished loop restarts as
`loop.abandoned_<i>` with its fit cache carried over, so the fits already made are reused.

# Run B (the rehearsal): status after one day, and one blocking fit

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

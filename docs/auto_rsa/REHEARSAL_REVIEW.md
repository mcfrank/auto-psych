# Simulated rehearsal: what it can and cannot tell us, and its unrecorded cost

From the local Sherlock session, 2026-10-09 ~10:00 PDT, for the PI and the driver session.
The rehearsal (`HANDOFF_sherlock_promote.md` step 4, array 47081376, code 0cc5624d) is
running. This note asks whether it is worth finishing, and records a cost-tracking gap.

## 1. Where it is (Fri 09:28, ~10 h in; no failures)

| task | selection | experiment 1 | experiment 2 | projected end |
|---|---|---|---|---|
| 1 `rehearsal_live` | fit on all data, select on live trials | done 23:32 → 06:42 (~7 h, 5 rounds) | inner loop round 2 (09:10) | ~19:00-21:00 Fri |
| 0 `rehearsal_all` | select on all data so far | done 23:31 → 08:55 (~9.4 h, 5 rounds) | design + simulated data done; scoring before its loop | ~01:00-03:00 Sat |

**Thread counts are fixed:** at 11 min, the parents had 8 threads and the fit children
6-7, each on its own core. The 48 h limit ends Sat 23:30.

**Experiment 1** (`private/experiment1/recovery.json`): ground truth `literal_listener`;
both tasks used the same 41 displays and 2,244 simulated trials.

| task | exported | RMSE to GT | closest live before → after |
|---|---|---|---|
| `rehearsal_all` | `rsa_l2_visual_confusion_fam_l0` | **0.095** | `oddity_heuristic_listener` 0.080 → `rsa_l2_max_confusion_fam_l0` 0.090 |
| `rehearsal_live` | `twin_contrast_solitary_oddity_listener` | **0.026** | 0.080 → 0.026 (the exported model) |

## 2. The caveat: the rehearsal's world is incoherent, so the comparison is decided by its setup

The rehearsal fits every model on two kinds of data:

- **existing data:** 50,587 real human trials (the five literature sources, train + test);
- **live data:** each experiment's 2,244 trials, simulated from the hidden ground truth.

The ground truth is `literal_listener` (L0), and **the human data reject L0 decisively**.
On run 2's held-out human conditions, L0 is **205 lpd behind `rsa_l2` (SE 82)** in all
three real cells, the worst of the five starting models by far.

So the rehearsal's "participants" behave in a way the literature says people don't, and
the existing and live data come from incompatible processes:

- **Cumulative selection** ranks models on ~96% human data after experiment 1 (~88% after
  three). It will keep preferring models that fit people, which move *away* from L0.
  Experiment 1 shows this: it ended farther from the ground truth (0.080 → 0.090) than it
  started.
- **Live-only selection** ranks on the L0-generated trials, so it moves toward L0
  (0.080 → 0.026).

That outcome is fixed by the construction, not learned from the run. **The rehearsal
cannot tell us which rule to use live.** In the live campaign, the existing and new data
both come from people, which is the opposite situation.

**Why the rule picked L0:** `src.rsa.outer.ground_truth` takes the starting model
farthest (design-pool RMSE) from its nearest promoted seed.

| starting model | nearest promoted seed | RMSE |
|---|---|---|
| `rsa_l1`, `rsa_l2` | themselves (they are in every chain) | 0 |
| `rsa_l1_salience` | `evaluative_prominence_l2` | 0.007 |
| `rsa_l1_shared_prior` | `evaluative_prominence_l2` | 0.010 |
| `literal_listener` | `lexical_preemption_heuristic` | 0.045 |

The only starting model the promoted seeds don't already reproduce is the one the data
reject.

**Also, L0 is odd for the paper.** A recovery claim of the form "the loop found the
mechanism that generated the data" is only interesting if that mechanism is a plausible
account of people. A demonstration in a world the prior literature contradicts is hard to
integrate with the existing-data analysis, which reports that L0 loses to every RSA model
(run 1 and run 2 results).

## 3. What the rehearsal does still validate

The **end-to-end machinery** of the live campaign, with no participants:

- experiment 1's design from the chain's seeds;
- simulated collection;
- prospective scoring;
- the inner loop on cumulative data (~52k trials) under the stop rule;
- export and carry-forward;
- experiment 2's design from the carried set;
- agents with no network, and one core per process (Research Computing's issue).

**Already shown:**
- task 1 has done experiment 1 → experiment 2 (carry-forward and a second design);
- task 0 has done experiment 1 and experiment 2's design;
- per-experiment wall time is **7-9.5 h**, mostly the inner loop on ~52k trials. That sets
  the live campaign's pace: one experiment per chain per working day, plus collection.

**Not yet shown:** a third experiment, and a full 3-experiment run end to end.

## 4. Cost gap: the outer loop records no agent spend

The rehearsal's inner loops write **no `token_usage.jsonl`**. Run 2's inner loop wrote one
per cell, and `rsa_status.sh` and `sweep_report` read it.

- Neither `_runs/outer/experiment*/model_loop/` nor the cell's `private/` holds one.
- The only trace is opencode's `[step] tokens=... cost=$...` lines in the job log. Those
  are cumulative within one agent call and interleave across the concurrent agents, so
  they can't be summed reliably.

**Estimate:** agent runs so far are 34 (task 0) and 46 (task 1); at run 2's $1.42 per run
that is roughly **$50-65 per task so far**. Finishing both tasks would be roughly another
$100-150, if experiments 2 and 3 resemble experiment 1.

**Fix (driver):**
- have `src.rsa.outer.run` pass the token recorder through to each experiment's inner
  loop, as `src.rsa.loop.run` does, so each `model_loop/` gets `token_usage.jsonl`;
- write a run-level summary (per experiment and total) beside them;
- add the cost to whatever status script the outer loop gets.

The live campaign needs this before it starts: it spends agent credits and Prolific money
together, and the budget in `PLAN.md` (~$720 for 3 chains × 3 experiments) can't be
checked without it.

## 5. The decision this was meant to inform, on principle

The rehearsal was to choose between two ways the live inner loop selects (PI 2026-10-09):
**cumulative** (rank on all data so far) or **live-only** (fit on all data, rank on the live
trials). Since the simulation can't decide it (§2), here is the case each way.

**For cumulative:**
- It is the Bayesian answer when the model class is adequate: the posterior over models
  uses all the data.
- It forces any exported model to account for the literature, which claim 2 relies on.

**For live-only** (fit on everything, rank on the new experiment):
- **Claim 2 is prospective.** The question is whether the loop's models predict *new*
  people in *designed* conditions better than the starting models, and the live trials
  are exactly that test. Selection on them is the out-of-sample test the paper reports.
- **The designed experiment exists to discriminate among the current models.** Its
  information lives in the new trials. With 50k existing vs 2.2k new trials, cumulative
  ranking is ~96% determined by the old conditions, which already produced the current
  set. A designed experiment would barely move selection.
- **Run 2 showed that fit to the training conditions does not reliably predict
  transfer.** In-sample and grouped-CV rankings disagreed with held-out ranks; for
  example, `feature_surprisal_listener` was 381.8 ELPD-CV behind its cell's winner yet
  ahead on held-out data.
- **Models are still fitted on all data,** so their parameters must accommodate the
  literature. Only the ranking is prospective.

**A middle rule worth considering:** fit on all data and rank on the live trials, but
refuse to export a model that falls clearly behind on the existing data (e.g. grouped CV
more than 2 × dse behind the best). The literature then constrains which models are
eligible, without drowning the new evidence.

The local session's view is that live-only (or the guarded middle rule) follows from the
paper's own claim and from run 2, and can be decided on principle without the rehearsal.
**That is the PI's decision.**

## 6. Options for the running jobs

1. **Cancel both now.**
   - Saves ~25 more node-hours of 8 cores and ~$100-150 of agents.
   - Loses the experiment 2 → 3 and 3-experiment end-to-end checks.
2. **Cancel task 0 (cumulative); let task 1 (live-only) finish.**
   - Task 0's remaining value is a selection comparison that §2 shows is predetermined,
     and it runs ~2 h slower.
   - Task 1 finishes the only end-to-end, multi-experiment test before real money is spent
     (~10 more hours, ~$40-70).
   - Its recovery numbers should be read as a mechanics check ("live-only selection
     follows the data that generated the new trials"), not as evidence about the selection
     rule.
3. **Let both finish** as planned. That costs the most, for a comparison that can't inform
   the decision.
4. **Redesign the rehearsal, if a simulation is still wanted, with a coherent ground
   truth:** one that fits the human data about as well as the best models, so existing
   and live data agree, but that the agents never see.
   - For example, a run-2 model that was admitted but not promoted, far from chain 0's
     seeds on the design pool, and within ~1 dse of the best on grouped CV.
   - The comparison would then ask a real question: does selection find a mechanism that
     is consistent with the literature, and that only the designed displays separate?
   - Check that nothing in the agents' tree names it (the promoted set is passed to the
     run with `--promoted`).

The local session recommends **option 2**, with the selection rule decided on principle
(§5). It will act only on the PI's instruction.

## 7. Outcome (2026-10-09, after the PI's decision)

**What happened:**
- **Task 0** (cumulative) was cancelled at 10 h 25 min, in experiment 2's design (PI).
- **Task 1** (live-only) was to be cancelled once experiment 3's prospective score was
  written. It **failed first**, at 15:08 (15 h 36 min).
- Outputs were brought back by hand into `data/rsa/rehearsal/<cell>/` (`outputs/` and
  `private/`, no trial CSVs, fit caches or agent logs), with `ground_truth.json`.

### Results

| task / experiment | exported (RMSE to GT) | closest live before → after | prospective `live_vs` best seed (= the GT) | `live_vs` best promoted |
|---|---|---|---|---|
| all, 1 | `rsa_l2_visual_confusion_fam_l0` (0.095) | 0.080 → 0.090 | -70.3 (SE 24.4) | -33.8 (SE 28.0) |
| live, 1 | `twin_contrast_solitary_oddity_listener` (0.026) | 0.080 → 0.026 | -70.3 (SE 24.4) | -33.8 (SE 28.0) |
| live, 2 | `twin_color_atten_base_valence_listener` (0.020) | 0.026 → 0.020 | -112.5 (SE 41.4) | **-100.0 (SE 42.2)** |

- Experiment 1's prospective score is the same in both tasks: the same models go in.
- The best promoted seed is `lexical_preemption_heuristic` (L0's nearest, 0.045). **The
  live models never beat it prospectively,** although by experiment 2 they are closer to
  the GT on the design pool (0.020-0.026). Design-pool RMSE and held-out lpd on the 40
  designed displays disagree here; worth understanding before reading claim 2's measure
  live.
- Experiment 1 excluded 13 of 200 simulated people for a catch-trial error
  (`max_catch_errors` 0): the ground truth with its fitted noise misses a catch trial
  6.5% of the time.
- Wall time per experiment (live): 7 h, then 7.5 h.

### Four things for the driver

1. **A reference model's fit timeout kills the run.** Experiment 3's prospective step
   refits every bar model on `prior.csv` (the five starting models, then every promoted
   seed).
   - `promoted:isolated_graded_costly_l3` hit the 30-min `FIT_TIME_LIMIT_SEC`. It had fit
     within the limit for experiment 2, with 2,268 fewer L0-generated trials.
   - The uncaught `FitTimeLimitExceeded` ended the run (`outer/run.py:234` →
     `loop/fitting.py:269`).
   - Live, this would hit *after* real data are collected. A resume would refit and likely
     time out again.
   - Suggest:
     - no time limit (or a much longer one) for bar and carried models: the SR pipeline's
       rule is "starting and carried models are never limited";
     - or record the model as unscored in `prospective.json` and go on.
2. **Experiment 3's design had nothing to discriminate.** The 12 models carried in were
   one lineage of near-duplicates (`twin_*`). Joint EIG reached only 0.23 of 3.58 bits
   (14 EIG picks before the noise floor), and power was **0.25**.
   - Under live-only selection the live set collapses to a single mechanism family, and
     the next design can't separate its members.
   - The live campaign needs diversity in what is carried: e.g. group the live set by
     prediction distance and carry the best of each group, as `promote` does. Or the
     design should treat near-duplicates as one hypothesis.
   - The novelty gate's 0.002 RMSE lets these variants through.
3. **stderr went to the checkout.**
   - `outer_rehearsal.sbatch` (and other RSA scripts) set `#SBATCH --error=%x_%A_%a.out`, a
     relative path. The handoffs' launch commands pass only `-o`, so tracebacks land in
     `~/auto-psych/` on Sherlock.
   - The rehearsal's error was only there. The checkout holds 11 such `.out` files.
   - Drop `--error` from the headers (stderr then follows `--output`), or pass `-e` too.
4. **Agent spend is still unrecorded** (§4).

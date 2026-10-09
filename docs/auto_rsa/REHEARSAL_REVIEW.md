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

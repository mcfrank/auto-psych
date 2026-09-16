# Merge plan: consolidating the September campaign, arm C, and main

*Drafted 2026-09-16 by Claude (Fable 5.1) for review. **Not yet executed.***
*Reviewers: please argue with §2 (what the evidence supports) before §4 (the
steps). If §2 is wrong the ordering does not matter.*

---

## 1. What exists and needs reconciling

Five separate lines of work, none merged:

| Line | Head | Contains |
|---|---|---|
| `main` | `9d5ccc1` | campaign + panel tooling, drop-path fix `884728e`, model guard `8c16a4e`, review-job mail `a112e12`, docs |
| campaign iter1 | `1f0abae` | PSIS-LOO exact-trial exemption |
| campaign iter2 | `bd7f032` | held-out label strip `85910cc`, export by ELPD rank `dfc84a6`, leakage audit `4f9f73e`+`bd7f032` |
| campaign iter3 | `ba8b2de` | carry live set + attempted-hypotheses ledger `9764f50`; **reverts** the leakage audit (`ba8b2de`) |
| campaign iter4 | `470e187` | candidate brief: `elpd_diff ± dse` **and** per-stimulus residual table |
| campaign iter5 | `2d450e2` | lens rotation **and** repair-instead-of-discard |
| arm C | `6ed41ea` | `raw_features` switch, `*_raw` seed sets, same-value collision rule, `pool_models_dir`, run verifier |

Each campaign branch is cumulative (iter4 contains iter1–3). Arm C branched from
iter2, so it does **not** contain iter3/iter4/iter5.

Also outside git: `recovery_improvement/recovery_2026_09_07/leakage_check_extension.patch`
(13,910 bytes) — the audit iter3 reverted.

## 2. What the evidence actually supports

All numbers are held-out RMSE (lower better) on paired cells, same `BASE_SEED`,
so repeat *r* is the same synthetic data on both sides. Pearson *r* is shown
only where it differs instructively.

| Sweep | falk_konold | finite_exp | motif_stack | local_rep |
|---|---|---|---|---|
| baseline | 0.045 | 0.014 | 0.147 | 0.103 |
| iter2 | **0.005** | **0.002** | 0.077 | 0.087 |
| iter3 | 0.008 | **0.002** | **0.062** | 0.095 |
| iter4 | 0.035 | 0.008 | 0.097 | **0.076** |
| iter5 | 0.046 | 0.010 | 0.107 | 0.084 |

**Claim 2a — iterations 1–2 produced the entire real gain.** RMSE falls ~9x on
`falk_konold_dp` and ~7x on `finite_experience_occurrence`. Both changes are
correctness fixes (a leaked answer; a tie broken by dictionary order), not
tuning, so the gain needs no statistical defence.

**Claim 2b — iteration 3 is the high-water mark and is safe.** Best `motif_stack`
RMSE of any sweep; ties iter2 on the easy two; mechanism-justified (stop
rediscovering models already tried).

**Claim 2c — iterations 4 and 5 regressed, and it is discovery, not selection.**
The *best model the loop ever held* worsened: 0.041 (iter3) → 0.050 (iter4) →
0.061 (iter5). Had the loop merely mis-chosen among good candidates, that column
would be flat and only the final would drift; the final-minus-best gap is
+0.0038 and +0.0010, an order of magnitude smaller than the shift in the best
column. 17% of iter4/iter5 cells ended on a carried *seed* rather than a
discovered model (0% before). Concretely, `falk_konold_dp` run3: iter3 ended on
`encoding_difficulty` at RMSE 0.004; iter4 ended on the `motif_stack` seed at
0.057.

**Claim 2d — Pearson *r* concealed 2c.** On that same cell *r* moved 1.000 →
0.975 while RMSE went 0.004 → 0.057. *r* is shift- and scale-invariant, so it
cannot see calibration error. Cells at *r* ≥ 0.9995 have RMSE 0.0005–0.011, i.e.
"perfect recovery" meant the right shape, not matching predictions.

**Claim 2e — the suspect is iteration 4's residual table, not its other half.**
It hands agents the incumbent's per-stimulus misses as z-scores, which invites
*patching the incumbent where it misses* (raising in-sample ELPD) rather than
proposing a different mechanism (generalising). Iteration 3 had just made every
rival within 2·dse persist, enlarging the pool being patched; iteration 5 then
kept rejected candidates alive too. This is the review panel's unresolved item
**U5**, logged before any of these sweeps ran. *Status: a hypothesis consistent
with the numbers, not a demonstrated cause — see §5.*

**Claim 2f — arm C is neutral on recovery.** Full 12 cells, paired: falk −0.005,
finite 0.000, motif_stack −0.036, local_rep +0.032. The early +0.234 I reported
was one cell of three. So the case for raw features is **principle** — the loop
demonstrably discovers the representation rather than reading it off supplied
columns — at no measurable cost. Caveat: that run logged **20 dropped models**
(candidates bound `occ_n20`, `multiscale_imbalance`, … which do not exist on raw
design rows), so its designs ran over a shrinking model set. It predates `884728e`.

**What is NOT supported:** that iter4 > iter3 or iter5 < iter4 by any margin
resolvable at 3 repeats when `local_representativeness` alone ranges 0.41–1.00.
Selection on the summary table is selection on noise — which is why §4 chooses
by mechanism and risk, not by score.

## 3. Decisions taken (and by whom)

1. **Base on iteration 3**, not 4. *(Claude, from 2b/2c.)*
2. **Keep arm C's machinery and make raw features the default.** *(User, on
   principle: a benchmark where the answer is reconstructible from supplied
   columns cannot distinguish discovery from regression. 2f says it is free.)*
3. **Split iter4 and iter5**, taking only the defensible half of each.
   *(Claude, from 2c/2e.)*
4. **Re-apply the leakage audit** that iter3 reverted. *(User-authored; iter3 was
   right by its own rules and wrong about provenance.)*

## 4. The plan

Ordering principle: **behaviourally inert changes first**, so anything that
breaks early is provably not a recovery regression; one commit per step with its
tests green before the next; the single combination that has never run is
verified at the end.

### Step 0 — Make the metric honest (before anything else)

**Do:** implement the panel's item 7 in the holdout reporter: held-out Bernoulli
log score and Brier primary, RMSE secondary, Pearson *r* descriptive; report
paired differences with bootstrap intervals over stimuli and repeats.
**Why:** 2d. Every step below is verified by a sweep; verifying on *r* would
repeat exactly the mistake that let two regressions look neutral for a week.
**Verify:** recompute the §2 table from existing `holdout.csv` files and confirm
it reproduces the RMSE ordering.
**Risk:** none to the loop; reporting only. Touches the protected evaluation
path, so it is user-side work, not a review agent's.

### Step 1 — Branch `consolidate/2026-09` from iteration 3

**Do:** `git fetch` iter3's branch into `main`'s repo; branch from `ba8b2de`.
**Why:** 2b. Inherits iter1+iter2+iter3 with no cherry-picking.
**Verify:** `pytest -q -m "not slow"` — expect the 18 known environmental
failures, nothing more.

### Step 2 — Merge `main`

**Do:** merge `9d5ccc1`. Expect conflicts in three files, all **complementary**;
resolve by keeping both sides, never by choosing one:
`src/models/pymc_inference.py` (main's `MissingStimulusColumns` vs arm-C-era
collision work), `src/pipelines/outer_loop/orchestrator.py` (main's
`screened_out_path` vs the design signature), `CLAUDE.md`.
**Why:** infrastructure and correctness with no effect on recovery. `884728e`
matters for step 5: it converts the silent drops that marred arm C into errors.
**Verify:** full fast suite at the 18-failure baseline.
**Risk:** low. Conflicts are textual, in regions I have already mapped.

### Step 3 — Re-apply the leakage audit

**Do:** `git apply` the preserved patch; commit with a message stating it is
user-authored and must not be reverted by a review agent.
**Why:** it is the only check that the two identity channels stay closed, and it
records each model's `pm.Data` column count — the input step 6 needs.
**Verify:** its own tests; confirm `any_csv_generating_model` and
`any_manifest_gt_named` appear in a fresh `holdout.json`.

### Step 4 — Split iteration 4: take the race, leave the residuals

**Do:** from `470e187` take only the `elpd_diff ± dse` presentation in
`pymc_orchestrator.py`; leave `incumbent_fit.py` (247 lines) and its two test
files out.
**Why:** 2e. Showing agents a real race instead of a misleading 1.000/0.000
softmax is justified a priori and is not the suspect; the residual table is.
**Verify:** the brief contains `elpd_diff`/`dse` and contains no per-stimulus
residual table; `test_parallel_candidates`, `test_pymc_inner_loop_*` green.
**Risk:** medium — a hand split of an agent-written commit. The seam looks clean
(different files), but check the presentation code does not import
`incumbent_fit`.

### Step 5 — Split iteration 5: take the lens rotation, leave the repair

**Do:** from `2d450e2` take only the lens-offset rotation; leave
repair-on-rejection and carried-model repair.
**Why:** lenses 3–6 never fired at all (iter4 tally: 72/72/72/0/0/0/0) — a plain
bug. Repair-on-rejection keeps weak candidates alive, which 2c/2e implicate.
**Verify:** `tests/test_lens_rotation.py` green; `tests/test_candidate_repair.py`
and `tests/test_carried_model_repair.py` absent, not failing. Confirm all seven
lenses appear in a `CANDIDATE_BRIEF.md` tally in the step-7 sweep.
**Risk:** **highest step.** One 656-line change to one file; the test files are
separate but the code may not be. If the split is not clean in an hour, take
neither half and record it as an open item rather than forcing it.

### Step 6 — Merge arm C and make raw features the default

**Do:** merge `6ed41ea`; set `raw_features: true` and `pool_models_dir:
.../seed_models_raw` in `holdout_recovery_faithful.yaml`; keep the featurized
seed set and config as the comparison arm.
**Why:** decision 3.2. `884728e` (step 2) now makes a design that cannot bind its
models fail loudly instead of quietly shrinking — the defect that marred arm C's
own run.
**Verify:** `SMOKE=1` one cell; then in step 7 confirm **zero** dropped models
and **zero** candidates importing the featurizer.
**Risk:** medium. Arm C branched from iter2, so this merge is where iter3's
carry-forward meets the `*_raw` seed sets; the ledger and the raw seeds touch
different files, but `pymc_orchestrator.py` is common ground.

### Step 7 — Verify the whole thing once

**Do:** one 12-cell sweep (3 repeats × 4 ground truths, `BASE_SEED=100`) from
the consolidated branch, `MAX_PARALLEL=12`.
**Why:** every step above is individually justified but **this combination has
never run**. Same seeds as iter3 makes it paired.
**Judge on:** step 0's metrics, primarily RMSE against iter3 (0.008 / 0.002 /
0.062 / 0.095). Expect ≈ iter3 on the three stable ground truths; `local_rep` is
uninterpretable at 3 repeats either way.
**Cost:** ~1.7 h/cell, ~2 h wall at 12-way, ~$165 Gemini.
**If it regresses:** step 4 and step 6 are the behavioural changes; bisect there.

### Step 8 — Settle `local_representativeness` separately

**Do:** 10 repeats of that ground truth alone (10 cells, ~2 h wall).
**Why:** it swings 0.41–1.00 and single-handedly made iters 3–5 look different
when they were not. Until its variance is characterised, no 3-repeat sweep can
resolve anything on it.

## 5. Open questions for reviewers

1. **Is 2e right?** The residual-table-induces-curve-fitting story fits the
   numbers but is not demonstrated. The direct test is the panel's U5 guard:
   per admitted model, provided-column count vs held-out change, and in-sample
   ELPD gain vs held-out change. The `agent_runs` archives hold the material.
   Should step 4 instead *keep* the residual table and add the guard?
2. **Is basing on iter3 over-conservative?** Iter4 has the best `local_rep` RMSE
   (0.076) of any sweep. If 2e is wrong, iter4 is the better base.
3. **Should step 5 be attempted at all**, given the split risk, when the benefit
   is one bug fix that could instead be rewritten from scratch in ~20 lines?
4. **Should raw features be the default, or the second arm?** Decision 3.2 takes
   it on principle; 2f says it is free but not better, and `motif_stack` is
   −0.036.
5. **Is there a cheaper verification than step 7?** E.g. 2 ground truths × 5
   repeats to get error bars on the stable cases instead of 4 × 3.

## 6. Known weaknesses of this plan

- Steps 4 and 5 are hand splits of agent-written commits. The author of this plan
  made several shell- and text-manipulation errors during this campaign; these
  two steps are the most error-prone and deserve the most scrutiny.
- The evidence base is 3 repeats per ground truth. Every claim about iters 3–5
  rests on differences smaller than the variance of one ground truth.
- Arm C's numbers come from a run with 20 dropped models. 2f may change after the
  step-7 re-run.
- No claim here is supported by more than one sweep per configuration.

# Amended merge plan: consolidating the September campaign, arm C, and main

*Drafted 2026-09-16 by Codex after reviewing `merge_plan_2026_09.md`, the
campaign repositories and archives, arm C's archived run, and disposable merge
preflights. **Not yet executed.***

*Reviewers: please challenge the corrected evidence in §2 and the raw-feature
acceptance gate in §5 before reviewing the mechanical merge order. The raw arm
that motivated the original default decision did not actually hide engineered
features from the candidate agents.*

---

## 1. Decisions this plan carries forward

1. **Use iteration 3 as the integration base.** It contains iterations 1–2's
   correctness fixes plus the live-set carry-forward and attempted-hypotheses
   ledger. This is a risk-based choice, not a claim that iteration 3 is
   statistically proven superior.
2. **Restore the user-authored leakage audit** reverted by iteration 3.
3. **Keep iteration 4's `elpd_diff ± dse` race presentation, but not its
   per-stimulus residual table.** Reimplement the retained part with its own
   tests rather than splitting the mixed commit mechanically.
4. **Keep iteration 5's lens rotation, but not repair-on-rejection or
   carried-model repair.** Reimplement the small scheduler directly.
5. **Merge arm C's machinery, but do not make it the default immediately.**
   First fix the inner-loop re-featurization bug, strengthen the verifier, and
   obtain a genuinely raw comparison run. If that gate passes, make the raw
   workflow the default in a separate commit while retaining an explicit
   featurized comparison configuration.
6. **Use held-out probability RMSE as the primary recovery metric.** Add
   expected Bernoulli KL regret as a secondary calibration-sensitive metric;
   keep Pearson correlation descriptive. Do not add Brier as a separate primary
   metric because its regret against a known probability target is exactly MSE.

The residual table and both repair mechanisms remain available as future
experiments. They are not part of this consolidation.

## 2. Corrected reading of the evidence

### 2.1 What the existing sweeps do support

- The robust observed recovery gains arrive by iteration 2. Those commits fix
  held-out-label leakage and selection/export correctness, so retaining them is
  justified independently of the small sweep sample.
- Iteration 3 is the preferred base because it preserves the easy-ground-truth
  gains, has the best observed `motif_stack` RMSE, sharply reduces
  `local_representativeness` variance in its three repeats, and implements a
  mechanism-level search improvement: live near-ties and the attempt ledger
  persist across experiments.
- Pearson correlation is insufficient as the primary metric. It concealed
  substantial shift/scale errors that RMSE exposed.
- Iterations 4 and 5 do not provide positive evidence for taking their mixed
  commits wholesale. The residual table and repair behavior both increase the
  risk of local patching, prompt growth, and continued investment in weak
  candidates.

### 2.2 Discovery versus selection is unresolved

The original plan called 0.041 → 0.050 → 0.061 the held-out RMSE of the "best
model the loop ever held." Those numbers are reproducible, but the label is too
strong: they are the minimum RMSE over the **selected incumbents recorded in the
trajectory**. `evaluate_trajectory` predicts several nonzero-weight models but
persists only the selected best model and the BMA. It does not persist held-out
scores for every admitted model in the zoo.

Therefore the existing statistic cannot distinguish:

- a discovery failure: no strong model was proposed/admitted; from
- a selection failure: a strong admitted model existed but never became the
  recorded incumbent.

Before making a causal claim about iteration 4, add an offline archive analysis
that scores every admitted model on the held-out pool and reports, per step and
cell:

1. the best held-out model available in the zoo (an oracle diagnostic only);
2. the incumbent selected by in-sample ELPD;
3. the final selected model; and
4. the oracle-minus-selected gap.

Until that analysis exists, describe the iteration-4/5 result as "worse
incumbent trajectories," not demonstrated discovery regression. The residual
table remains excluded on prospective overfitting risk, cost, and lack of
positive evidence—not because it has been proved causal.

### 2.3 Same seed does not mean the same synthetic data

`BASE_SEED=100` matches stochastic seeds across configurations. Experiment 1
can therefore be identical when its initial design is identical. Later
experiments are adaptive: different candidates and posteriors produce different
designs, so the stimuli and generated response files diverge. The held-out pool
also differs slightly because each run excludes its own training pairs.

Use the term **matched-seed cells**, not "the same synthetic data." For a
pairwise held-out comparison, re-score both cells on one common evaluation pool:
the exhaustive stimulus universe minus the **union** of the two cells' training
pairs.

### 2.4 Arm C's recovery result is invalid for its intended question

Arm C correctly wrote a five-column raw file at
`experimentN/data/responses.csv`, and its EIG design used raw stimulus rows.
However, before running the inner loop,
`run_inner_model_loop_programmatic()` pooled those raw files and unconditionally
re-applied the project featurizer. The archived arm-C run confirms:

- `experiment1/data/responses.csv` has the five raw columns;
- `experiment1/model_loop/responses.csv` has 60 columns, including the
  engineered features; and
- the candidate `CONTEXT.md` explicitly lists those engineered columns.

The verifier checked only `experimentN/data/responses.csv`, so it reported the
data as raw while missing the actual candidate-facing CSV. The 20 EIG drops are
consistent with this failure: agents built models against supplied feature
columns, then those models could not bind to raw design rows.

Consequences:

- Arm C does **not** demonstrate that agents rediscovered the representation
  from raw sequences.
- Its recovery deltas cannot justify "raw features are free."
- `884728e` will make the same mismatch fail loudly, but does not itself fix the
  mismatch.
- A genuinely raw run is required before changing the default.

## 3. Metrics and comparison protocol

### 3.1 Per-cell metrics

For ground-truth probabilities `q_i` and recovered probabilities `p_i` on the
common held-out pool:

1. **Primary — probability RMSE**

   `sqrt(mean((p_i - q_i)^2))`

2. **Secondary — expected Bernoulli KL regret**

   `mean(q_i log(q_i/p_i) + (1-q_i) log((1-q_i)/(1-p_i)))`

   Clip `p_i` and `q_i` to a documented epsilon (proposed `1e-9`) for numerical
   stability. Reporting regret rather than raw cross-entropy puts the oracle at
   zero and makes comparisons interpretable.

3. **Descriptive — Pearson correlation**, bias, and calibration slope/intercept.

Expected Brier regret is `(p_i-q_i)^2`; averaged over stimuli it is MSE, so it
should not be presented as independent evidence alongside RMSE.

### 3.2 Historical recomputation

Existing `holdout.csv` files contain aggregate RMSE/correlation only. They are
insufficient to derive KL regret or a common-pool comparison. Re-evaluate from
the archived model sources and MCMC caches, or explicitly record that a legacy
cell has only the old metrics. Do not silently manufacture new columns from the
aggregate CSV.

### 3.3 Uncertainty

The exhaustive held-out stimulus space is the benchmark target, not an IID
sample from which stimulus-level uncertainty should be inferred. Stimulus pairs
also share component sequences and are strongly dependent. Do not bootstrap
individual stimuli and present the resulting narrow interval as uncertainty in
loop performance.

The stochastic unit is the repeat/run. Report every matched-seed difference,
the mean and median difference, and dispersion across repeats. With only three
repeats, treat intervals as descriptive; do not use them to claim equivalence.
If a formal interval is desired, add repeats and resample paired repeats only.

### 3.4 Predeclared practical margin

Before launching the comparison, record a smallest practically important
degradation for probability RMSE. Proposed value: **0.02 RMSE** on a
probability scale. This is a decision threshold, not a significance cutoff.
Changing it after seeing the sweep requires an explicit amendment.

## 4. Exact inputs and preflight

### Step 0 — Freeze and verify the inputs

**Do:** create a fresh integration checkout/worktree rather than operating in a
dirty research checkout. Load the Git version used by the campaign if needed.
Fetch these local branches under explicit integration refs and verify their full
SHAs:

| Input | Expected head before review amendments |
|---|---|
| campaign iteration 3 | `ba8b2de66c45305b66066545ed8cf30c5cad208f` |
| campaign iteration 4 reference only | `470e187ed2beded050c267830380d14f1e02d27c` |
| campaign iteration 5 reference only | `2d450e21f20c5748b870e479f859766fd47b5b69` |
| arm C | `6ed41ea6d8994473eab0b1b79959d9ec706eb340` |
| main | execution-time immutable SHA; currently `6ec013e` plus any commit carrying this amended plan |

The fallback leakage patch is:

`$SCRATCH/auto-psych/recovery_improvement/recovery_2026_09_07/leakage_check_extension.patch`

with SHA-256:

`cf3be3eed6373dda6192e3729971c47fd2884afc44651e511cc686b7b0b2764c`

It should not be needed if Step 3 uses `git revert ba8b2de`.

**Record:** `git status`, all resolved SHAs, Python/dependency lock state, agent
backend/model string, and the exact list of baseline failing test node IDs.
Comparing only a count such as "18 failures" is unsafe because one old failure
could disappear while one new failure appears.

**Verify:** every source checkout is clean; `git merge-base` confirms arm C
descends from iteration 2 and iterations 4–5 descend from iteration 3.

## 5. Integration sequence

One commit per numbered behavioral or infrastructure step. Tag or otherwise
record each checkpoint so a sweep can be tied to an exact tree.

### Step 1 — Branch from iteration 3

**Do:** create `consolidate/2026-09` at `ba8b2de`.

**Verify:** run the targeted campaign tests and the full fast suite. Store the
exact pass/fail node list as the iteration-3 baseline.

### Step 2 — Merge current main

**Do:** merge the frozen execution-time `main` SHA, not the stale `9d5ccc1`
listed in the original plan.

**Preflight result:** iteration 3 merged with `6ec013e` cleanly in a disposable
clone; `src/models/pymc_inference.py`,
`src/pipelines/outer_loop/orchestrator.py`, and `CLAUDE.md` auto-merged. Re-run
the preflight against the final main SHA rather than assuming this remains true.

**Verify:** targeted tests for `MissingStimulusColumns`, design screening, model
selection, campaign review infrastructure, then the full fast suite. No new
unexpected failing node IDs.

### Step 3 — Restore the leakage audit by reverting the revert

**Do:** run `git revert ba8b2de` and give the new commit a message explaining
that the restored audit is user-authored and protected. Prefer this to applying
an untracked patch: `ba8b2de` is an explicit, single-purpose revert of the audit
and its tests.

**Verify:**

- all restored leakage-audit unit tests pass;
- positive fixtures flag a `generating_model` CSV header and a seed manifest
  naming the held-out model;
- negative fixtures ignore loop-output manifests; and
- per-model `pm.Data` columns are recorded.

Defer the expensive "fresh holdout.json" check to the smoke run in Step 9.

### Step 4 — Merge arm C as opt-in machinery only

**Do:** merge `6ed41ea`, retaining the raw seed directories, collision rule,
raw config, pool selection, and verifier. Do **not** edit
`holdout_recovery_faithful.yaml` or switch any launcher default in this commit.

**Preflight result:** after Steps 2–3, the current refs conflict only in
`CLAUDE.md`. Without the audit restoration, the arm also conflicts in the
holdout-recovery test file. Re-run this preflight after the final preceding
commits. The three-conflict list in the original plan belongs to an older
main-versus-arm merge and is not the observed conflict set for this sequence.

**Verify:** raw seed parity tests, collision tests, pool-seeding tests, and the
full fast suite. Confirm that featurized-mode outputs are byte/schema compatible
where expected because raw mode is still opt-in.

### Step 5 — Make raw mode genuinely raw end to end

**Do:** add `raw_features: bool = False` to
`run_inner_model_loop_programmatic()` and thread it from the holdout config and
runner. When true:

1. pool the raw response rows without applying the project featurizer;
2. write a five-column `model_loop/responses.csv`;
3. ensure candidate and critique contexts advertise only those five columns;
4. require seed and admitted models to compute any derived stimulus features
   themselves; and
5. validate candidate portability on representative raw stimulus rows before a
   model can be carried into a later EIG design.

Add configuration validation so raw mode cannot start unless both model pools
are the raw variants:

```yaml
raw_features: true
seed_models_dir: src/subjective_randomness/pymc_model_families_raw
pool_models_dir: src/pipelines/outer_loop/projects/subjective_randomness/seed_models_raw
```

Prefer capability validation—models can bind raw rows—over relying only on a
directory-name suffix.

**Strengthen the verifier:**

- check every `experiment*/data/responses.csv` **and**
  `experiment*/model_loop/responses.csv`, including archived runs;
- inspect candidate `CONTEXT.md` column declarations;
- fail, rather than warn, if no candidate-facing CSV was found;
- require zero `[drop]` lines and zero `screened_out.json` model exclusions;
- reject or fail the run if candidate source imports the project featurizer;
- confirm all configured cells completed; and
- report the resolved config, code SHA, and feature mode.

**Tests:** add a regression test that would reproduce arm C's bug: raw
`data/responses.csv` followed by an incorrectly featurized
`model_loop/responses.csv` must fail. Add an end-to-end fixture asserting the
candidate prompt contains no engineered columns.

### Step 6 — Add honest reporting and offline diagnostics

**Do:** add RMSE, KL regret, bias, calibration slope/intercept, and descriptive
Pearson reporting for future runs. Preserve old CSV columns for downstream
compatibility and version the output schema.

Add a separate offline analysis command that:

- re-scores two matched-seed cells on the common held-out pool;
- evaluates every admitted model for the discovery-versus-selection diagnostic;
- emits per-cell values rather than only aggregate means; and
- records cells that cannot be reconstructed from their archives.

This is user-side evaluation code. It must not be available to candidate or
review agents during a run.

**Verify:** analytic unit tests for RMSE and KL, clipping boundary tests,
backward-compatible CSV/JSON reader tests, and a small cached-run reanalysis.
Confirm historical RMSE remains unchanged; do not require new metrics to be
recoverable from aggregate legacy CSVs.

### Step 7 — Reimplement the iteration-4 race presentation

**Do:** change `existing_hypotheses.md` so live models are shown best-first with
rank, `elpd_diff ± dse`, and PSIS-LOO reliability instead of a rounded softmax.
Use "not clearly separated at approximately 2·dse" rather than claiming a
formal statistical tie.

Do not import or add `incumbent_fit.py`; do not expose per-stimulus residuals,
fitted parameter values, or incumbent source code.

**Why reimplement:** `470e187` changes both behaviors inside
`pymc_orchestrator.py`, and its candidate-brief tests cover the combined path.
It is not a clean path-level cherry-pick.

**Verify:** add a dedicated race-presentation test covering ordering, margins,
reliability warnings, a missing comparison row, and the absence of residual
content. Run existing parallel-candidate, history, critique, and ledger tests.

### Step 8 — Reimplement lens rotation

**Do:** implement a small pure lens scheduler whose index is:

`(experiment_offset + iteration * candidate_count + candidate_idx) % n_lenses`

Continue the offset across experiments so all seven default lenses fire over a
standard 3-experiment × 2-round × 3-candidate run. Record the chosen lens in the
ledger.

Do not bring over `AdmissionOutcome`, candidate repair, carried-model repair, or
the `candidate_repairs` knob from `2d450e2`.

**Why reimplement:** the desired behavior is small, while the source commit
mixes it into a 656-line orchestrator change. Adapt the lens test so it does not
depend on iteration 4's omitted `incumbent_fit` fixtures.

**Verify:** pure schedule coverage/balance tests, prompt-to-ledger consistency,
outer-to-inner experiment offset threading, and confirmation that no repair
files, calls, or configuration knobs exist.

### Step 9 — Static checks, full tests, and two smoke runs

Run after all integrations, before any full sweep:

1. `git diff --check` and Python compile checks.
2. All targeted tests named above.
3. Full fast suite, compared by exact failure node ID with the stored baseline.
4. One featurized smoke cell.
5. One true-raw smoke cell.

The raw smoke is accepted only if:

- both response CSV layers and candidate contexts are raw-only;
- all seeds and carried candidates bind to raw design rows;
- zero models are dropped or screened out;
- no candidate imports the project featurizer;
- the leakage audit fields appear in `holdout.json`;
- all seven lenses appear when the smoke has enough slots, otherwise a
  deterministic test proves the full schedule; and
- the run reaches final evaluation without a traceback.

Any failure blocks the comparison sweep. Do not reinterpret a partial raw run
as scientific evidence.

## 6. Same-commit comparison before changing the default

### Step 10 — Run featurized and true-raw arms from one commit

Run two 12-cell sweeps from the exact Step-9 commit:

- 3 repeats × 4 ground truths, featurized config;
- 3 repeats × 4 ground truths, true-raw config;
- `BASE_SEED=100` for both; and
- identical agent backend/model, sampling settings, prompt code, and scheduling
  policy.

Launch them as a balanced block with a shared total concurrency cap so one arm
does not run under systematically different rate limits or cluster contention.
Record job IDs, resolved configs, dependency locks, code SHA, and agent model
identifier. These are matched-seed—not identical-data—comparisons.

Estimated cost is approximately twice a 12-cell sweep. This replaces the
original plan's 12-cell all-on sweep plus automatic 10-cell
`local_representativeness` extension; it is only modestly more expensive than
that combined budget and gives a valid same-code control arm. Do not schedule
extra local-representativeness repeats automatically; decide after inspecting
the paired result.

### Step 11 — Re-score on common pools and apply the gates

For every matched cell pair, build the exhaustive common pool excluding the
union of both training sets and compute the metrics in §3. Report individual
paired deltas before means.

**Engineering gate (mandatory):**

- all 24 cells complete;
- both verifiers pass;
- raw candidate-facing data is truly raw;
- zero silent or explicit model drops;
- no featurizer imports; and
- no leakage-audit failure.

**Practical recovery gate for switching the default:**

- no stable ground truth (`falk_konold_dp`,
  `finite_experience_occurrence`, `motif_stack`) has mean raw-minus-featurized
  RMSE worse than the predeclared 0.02 margin;
- no stable ground truth shows a practically important loss in all three paired
  repeats; and
- `local_representativeness` is reported cell by cell and is not used alone to
  overrule the stable-ground-truth result at n=3.

This gate is deliberately practical rather than a claim of statistical
equivalence. If it fails, raw mode remains a maintained second arm and the
default stays featurized while the failure is diagnosed.

Also run the all-model oracle diagnostic. If the chosen incumbent regresses but
the best admitted model does not, investigate selection. If both regress,
investigate search/discovery. This determines where a follow-up ablation belongs.

### Step 12 — Switch the default only after the gate passes

If Steps 10–11 pass, make raw mode the workflow default in a standalone commit.
Do not erase the historical featurized configuration or silently change the
meaning of an old config path.

Recommended layout:

- retain/version an explicit featurized config;
- retain/version an explicit raw config;
- point the standard launcher/default alias at the raw config; and
- write `feature_mode`, both resolved seed directories, and the config digest
  into every result.

Update the decision record with the corrected arm-C bug, the clean comparison,
and the exact gate result.

If the gate fails, record that outcome and merge the raw machinery as opt-in
without the default-switch commit.

## 7. Optional follow-ups, not blockers for consolidation

1. **Residual-table causal test:** run the U5 analysis correlating provided
   columns and in-sample ELPD gain with held-out change. Only then consider a
   controlled residual-table arm.
2. **Repair behavior:** split timeout/file salvage from mechanism repair before
   testing either. They have different risk profiles and should not share one
   switch.
3. **More `local_representativeness` repeats:** if the same-commit comparison
   remains ambiguous, add matched raw/featurized pairs sequentially. Do not run
   ten repeats of only one arm when the decision concerns the difference
   between arms.
4. **Import isolation:** static rejection of featurizer imports is a minimum.
   Process-level isolation would make the raw benchmark stronger because merely
   copying featurizer source remains possible while the repository is readable.

## 8. Stop conditions

Stop and amend the plan rather than forcing the merge when any of these occurs:

- a supposedly raw candidate sees an engineered feature column;
- a raw model reaches design only to fail binding;
- a source branch SHA differs from the frozen input without explanation;
- the exact fast-suite failure set gains a new unexplained failure;
- the all-model archive analysis cannot reconstruct enough cells to support a
  discovery-versus-selection claim; or
- a practical acceptance threshold is changed after looking at sweep results.

## 9. Expected final history

The intended integration history is conceptually:

1. iteration-3 base;
2. current-main merge;
3. restore leakage audit;
4. merge arm-C machinery, still opt-in;
5. true-raw inner-loop and verifier fix;
6. reporting/common-pool/all-model diagnostics;
7. race presentation only;
8. lens rotation only;
9. test/smoke fixes, if any; and
10. raw-default switch **only if** the same-commit comparison passes.

This structure keeps infrastructure, prompt behavior, and the benchmark-default
decision separately reviewable and revertible.

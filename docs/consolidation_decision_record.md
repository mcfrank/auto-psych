# Consolidation decision record — September 2026

This document records the decisions, evidence, and open questions from the
September 2026 consolidation of five unmerged lines of work into the
`consolidate/2026-09` branch.

## Inputs and SHAs

| Input | SHA | Role |
|---|---|---|
| Campaign iteration 3 (base) | `ba8b2de66c45305b66066545ed8cf30c5cad208f` | Branch base; carries iterations 1-2 correctness fixes, live-set carry-forward, attempted-hypotheses ledger |
| Campaign iteration 4 (reference) | `470e187ed2beded050c267830380d14f1e02d27c` | Race presentation reimplemented from here; code not cherry-picked |
| Campaign iteration 5 (reference) | `2d450e21f20c5748b870e479f859766fd47b5b69` | Seven-lens rotation reimplemented from here; code not cherry-picked |
| Arm C | `6ed41ea6d8994473eab0b1b79959d9ec706eb340` | Raw-features machinery merged; branched from iteration 2, does not contain iteration 3 |
| `main` at consolidation time | `3685aeb6dd3ec17e82590455fe0d3b08fc10fc67` | Infrastructure: `MissingStimulusColumns`, campaign/panel tooling, docs |
| Leakage-audit patch | sha256 `cf3be3eed6373dda6192e3729971c47fd2884afc44651e511cc686b7b0b2764c` | Fallback for restoring the user-authored leakage audit |

## Decisions

### 1. Base on iteration 3, not `main` or arm C

Iteration 3 carries the correctness fixes from iterations 1-2 (held-out label
leak, export by ELPD rank instead of softmax argmax, PSIS-LOO exemption for
constant-log-likelihood trials) plus the live-set carry-forward and the
attempted-hypotheses ledger. These are the most consequential behavioural
changes. `main` has infrastructure that iteration 3 lacks; arm C has raw-mode
machinery that iteration 3 lacks. Both are merged on top.

### 2. Merge `main` for infrastructure, not behaviour

`main`'s `MissingStimulusColumns` fail-loud screen (`884728e`), campaign/panel
tooling, and documentation are merged. No behavioural changes from `main` conflict
with iteration 3.

### 3. Restore the user-authored leakage audit

Iteration 3 reverted the leakage audit (commit `ba8b2de`); this consolidation
re-reverts it. The audit is user-authored and tests that no held-out information
leaks into the agent's observable artefacts: no CSV-generating ground-truth model
in the manifest, no ground-truth name in the manifest, no identical model file.
These tests run on every recovery cell.

### 4. Merge arm C as opt-in machinery

Arm C's additions are kept: `loo_reliability.py`, the same-value collision rule
(`_same_feature_value` / `PROTECTED_ROW_COLUMNS` in `pymc_inference.py`),
`pool_models_dir`, raw seed sets (`pymc_model_families_raw/`, `seed_models_raw/`),
`remove_manifest_entry.py` and the array script's manifest scrub, and the verifier.

Arm C's _behaviour_ (iteration 2's prune rule, no ledger, no live-set carry)
is not kept; iteration 3's behaviour is the base.

### 5. Make raw mode genuinely raw end to end

In a `raw_features: true` run, no agent-facing artefact carries an engineered
column. `RAW_RESPONSE_COLUMNS` is the single source of truth for the five raw
columns. Config validation checks that every model in the pool and seed manifests
can bind a raw stimulus row. The verifier checks every agent-facing CSV, candidate
context, and candidate source in a finished run.

### 6. Uniform design prior and dse-only prune rule (iteration 3)

Iteration 3 (`9764f50`) replaced the stacking-weight registry with a uniform
prior over the carried set and removed the prune weight floor (prune on
`elpd_diff > dse_multiplier * dse` alone). Rationale: stacking weights are
ensemble coefficients rather than plausibility, and in 15 of 40 next-experiment
designs of the iteration-2 recovery sweep, every model actually present had
weight ~0 (or one had 1.0), so all 32 EIG-selected stimuli had zero EIG. The
stacking weights remain as a report field in `model_posterior.json`.

### 7. Reimplement, not cherry-pick, from iterations 4 and 5

The race presentation (iteration 4) and seven-lens rotation (iteration 5)
are reimplemented with fresh tests. What is kept: candidates see models ranked
by ELPD-LOO with `elpd_diff +/- dse` and a separation verdict; lenses rotate
with `(offset + iteration * candidate_count + candidate_idx) % n_lenses` so
all seven fire. What is deliberately not brought over: `incumbent_fit.py`,
per-stimulus residual tables, `AdmissionOutcome`, candidate repair,
carried-model repair, and the `candidate_repairs` knob.

### 8. Honest metrics

RMSE is primary, expected Bernoulli KL regret is secondary, Pearson r is
descriptive. Brier regret against a known probability is `(p - q)^2` (i.e. MSE),
so it is not a second metric. Per-repeat paired differences on a common held-out
pool, no stimulus bootstrap. `evaluate_trajectory` rows gain `kl_regret`, `bias`,
`calib_slope`, `calib_intercept` and BMA variants.

## The arm C finding: 5-vs-59-column evidence

Arm C (`6ed41ea`) was intended as a raw-features run. In the archived run
(`$ARMC_RUN_ROOT/run1/falk_konold_dp/agent_runs.tar.gz`):

- `experiment1/data/responses.csv` has 5 columns (raw sequences + response).
- `experiment1/model_loop/responses.csv` has 59 columns (all engineered features).
- Every candidate `CONTEXT.md` lists the engineered columns.

The cause: `run_inner_model_loop_programmatic` on `6ed41ea` takes no
`raw_features` parameter and always calls `_load_project_featurizer` then
`_write_feature_csv`. The arm C verifier only checked `data/responses.csv`, not
`model_loop/responses.csv`. The 20 `[drop]` lines in the run are candidates
written against supplied engineered columns that could not bind raw design rows.

This consolidation fixes the gap: `run_inner_model_loop_programmatic` accepts
`raw_features: bool = False` and, when true, skips the featurizer entirely;
the verifier checks both CSV layers and candidate contexts.

## Metric protocol

For ground-truth probabilities `q_i` and recovered `p_i` on the common held-out
pool:

- **Primary**: `RMSE = sqrt(mean((p_i - q_i)^2))`
- **Secondary**: `KL regret = mean(q_i * log(q_i / p_i) + (1 - q_i) * log((1 - q_i) / (1 - p_i)))` with `p`, `q` clipped to `[1e-9, 1 - 1e-9]`
- **Descriptive**: Pearson r, bias `mean(p_i - q_i)`, calibration slope and intercept (OLS of p on q)

The repeat is the stochastic unit. Report every matched-seed paired delta, then
mean/median/min/max per ground truth. No stimulus bootstrap. Predeclared
practical margin: **0.02 RMSE**. Changing it after seeing results needs a written
amendment.

## The raw-vs-featurized gate

The default switch to raw needs a featurized arm from the same commit, run as a
balanced block (shared concurrency, same backend and model, same `BASE_SEED=100`,
the same repeats).

**Engineering gate (mandatory):** all cells complete; both verifiers pass; raw
candidate-facing data truly raw; zero drops; no featurizer imports; no
leakage-audit failure.

**Practical recovery gate:** on the stable ground truths (`falk_konold_dp`,
`finite_experience_occurrence`, `motif_stack`) the mean raw-featurized RMSE delta
is not worse than +0.02, and no stable ground truth is worse in all paired
repeats. `local_representativeness` is reported cell by cell and does not decide
alone. If the gate passes, the default switch is its own commit. If it fails, raw
stays a maintained opt-in arm.

## The manifest-scrub confound

`main`'s `holdout_recovery_array.sbatch` deletes the held-out `.py` from the
registry and the pool and stubs the family twin, but leaves the
`models_manifest.yaml` entry (name + rationale). Arm C's array script removes
that entry (`remove_manifest_entry.py`). Merging arm C therefore closes this
information channel for every future sweep, featurized or raw. This is a
deliberate confound against iteration 3's numbers: any comparison between
iteration 3's sweep and the consolidated sweep includes both the code changes
_and_ the manifest scrub. All such comparisons are labelled **confounded** in
the results.

## What remains open

1. **Residual-table causal test.** Iteration 4's per-stimulus residual table was
   not included because its benefit is unestablished. A controlled experiment
   (same commit, residual table on vs off) would determine whether it helps.

2. **Candidate repair behaviour.** Iterations 4-5 introduced `AdmissionOutcome`
   and candidate repair (re-running a rejected candidate with the rejection reason
   injected). This was not included because it couples admission and generation
   and makes the novelty gate less clean. It may be worth revisiting if discovery
   failure rates are high.

3. **Process-level import isolation.** Agent-written candidates can import the
   project featurizer at runtime. The verifier catches this after the fact by
   grepping source files. Process-level isolation (running candidate fits in a
   subprocess with a restricted import path) would prevent it structurally.

4. **Featurized arm from the same commit.** The raw-vs-featurized gate (above)
   requires a featurized arm from the consolidated commit. This is not part of the
   current sweep (`SWEEP_ARMS=raw`); the user may run it separately.

## History of specific choices

Empirical observations that motivated specific constants, rules and designs.
Moved here from inline source comments during P25 so the code carries only
the "why" while the provenance stays recorded.

### Export by ELPD-LOO rank, not softmax argmax (`scoring._best_exportable_model`)

The softmax posterior is rounded to six decimals, so every model more than ~14
nats behind the argmax reads 0.0 and ties. A `max` over those ties returned
whichever model came first in the manifest. In the pre-campaign baseline sweeps,
62 of 230 experiments exported a far-behind seed model this way — hundreds of
nats behind a reliable agent-written model. The fix: select by `az.compare`'s
ELPD-LOO rank among PSIS-LOO-reliable rows.

### The hypothesis ledger (`hypothesis_ledger.py`)

Without the ledger, pruned hypotheses disappeared from `existing_hypotheses.md`
and were re-proposed by the next round's candidate agents. In the iteration-2
recovery sweep, 15% of candidate slots re-proposed a name already tried in the
same cell; in the weakest cell 11 of 13 re-proposals had already been pruned
there. The ledger records every event and renders into every candidate brief as
the "already tried — do not re-propose" section.

### Novelty RMSE threshold 0.02 (`model_zoo.DEFAULT_NOVELTY_RMSE_THRESHOLD`)

0.02 sits just below the closest genuinely-distinct pair observed across the
human replicates (run 2's two winners, RMSE 0.029). Below this, two models'
posterior-mean predictions are empirically indistinguishable on the observed
stimuli.

### Stacking weights are not plausibility (`model_zoo._prune_losers`)

Pruning uses `elpd_diff > multiplier * dse`, not stacking weights, because
`az.compare`'s weights are ensemble coefficients: a model 1.6 nats behind the
best can read 0.000 (its predictions are redundant with the best's) while one
95 nats behind can read 0.33 (they differ). The weight floor was removed in
iteration 3 for this reason.

### Seven-lens rotation (`candidate_agent.DEFAULT_CANDIDATE_HINTS`)

The old three-hint rotation pushed genuine novelty in only one candidate of
three; the remaining two hints encouraged conservative revision. The seven-lens
battery assigns each candidate a distinct exploration strategy so the full
hypothesis space is covered more evenly across rounds.

### MCMC sampler defaults (`mcmc_defaults.py`)

Before the centralized defaults module, sampler settings had drifted per entry
point (outer 2000/2000/4, inner CLI 500/500/2, PPC 2000/2000/4, design twins
hard-coded 500/500/2). `draws` and `tune` were raised, and `target_accept`
was lifted from PyMC's implicit 0.8 default to 0.99 to shrink divergences and
stabilize the PSIS-LOO tail. (2026-08-13 audit: the hard export gate on
Pareto-k no longer exists — the only hard ELPD gates today are on non-finite
values — so 0.99 stands as cautious tail stabilization.) `cores` was changed
from 1 (four production chains ran sequentially for no stated reason) to 4;
measured 2026-08-13 on macOS, chains=4, PyMC 5.28.5: cores=4 finished in 38 s
vs 110 s at cores=1, with no multiprocessing trouble.

### PSIS-LOO exact-trial exemption (`loo_reliability.py`)

Measured on the 2026-08/09 holdout sweeps, every non-finite k in the excluded
winners' fits was a constant-log-likelihood trial (40–720 per model, one per
participant × stimulus), and not one trial had a finite k above 0.7. Those
winners were dropped for nothing.

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
carried-model repair, and the `candidate_repairs` knob. (Candidate repair was
later reintroduced in a bounded form by P36 of the loop-improvement plan —
see "One retry and one repair per candidate slot" below.)

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
   failure rates are high. *Resolved (September 2026, P36):* discovery failure
   rates were high — see "One retry and one repair per candidate slot" below.

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

### Novelty RMSE threshold 0.02 (`model_zoo.DEFAULT_NOVELTY_RMSE_THRESHOLD`) — superseded

0.02 sat just below the closest genuinely-distinct pair observed across the
human replicates (run 2's two winners, RMSE 0.029). Below this, two models'
posterior-mean predictions were taken to be empirically indistinguishable on
the observed stimuli. Superseded by the entry below (September 2026).

### Novelty gate on a loop-generated pool at 0.002 (`model_zoo.novelty_pool_rows`, `DEFAULT_NOVELTY_RMSE_THRESHOLD`)

The gate compared posterior-mean `p_left` on the 64 training stimuli. The 23
rejection margins archived across the September 2026 sweep (`sweep_rerun`)
were bimodal:

```
0.0000 x2  0.0001  0.0002  0.0004  0.0018  0.0031  0.0036  0.0061  0.0088
0.0097 x2  0.0114  0.0115  0.0117  0.0130  0.0133  0.0152  0.0166  0.0169
0.0173  0.0183  0.0191
```

About five genuine re-skins cluster at ~0 (two predicted *identically*: a
byte-identical model file shares the cached fit), and about eighteen spread
evenly from 0.006 up to the threshold — what distinct mechanisms that happen
to agree on 64 points look like. Two changes:

- **Where.** `_min_prediction_rmse` now compares predictions on a pool the
  loop generates from its own seed (`NOVELTY_POOL_SEED`, 512 same-length
  pairs at lengths 4–8, the design's pair universe) and records as
  `model_loop/novelty_pool.json`. It is deliberately not the recovery
  harness's eval pool: the loop must not select models on the stimuli it is
  later scored against. Two mechanisms that agree on the training stimuli but
  not elsewhere are now told apart.
- **Threshold.** 0.002 sits in the gap of the distribution above. Removing
  the gate instead is not an option: a model predicting identically to the
  incumbent is statistically tied with it (`elpd_diff ≈ 0 < 2·dse`), so
  pruning never removes it, it is carried forward, refit every scoring pass,
  and fills the briefs with near-identical entries.

Caveat: the margins above were measured on the training stimuli with the
full posterior, where the posterior is tightest; on a pool off the training
data the Monte-Carlo noise in two independent fits' posterior means is
larger, so the gate predicts on every draw (no thinning) to keep that floor
well below 0.002. The threshold is a knob (`inner_loop.novelty_rmse_threshold`
in the holdout config) so it can be A/B'd.

### Stacking weights are not plausibility (`model_zoo._prune_losers`)

Pruning uses `elpd_diff > multiplier * dse`, not stacking weights, because
`az.compare`'s weights are ensemble coefficients: a model 1.6 nats behind the
best can read 0.000 (its predictions are redundant with the best's) while one
95 nats behind can read 0.33 (they differ). The weight floor was removed in
iteration 3 for this reason.

### Twelve-lens rotation (`candidate_agent.DEFAULT_CANDIDATE_HINTS`)

The old three-hint rotation pushed genuine novelty in only one candidate of
three; the remaining two hints encouraged conservative revision. The seven-lens
battery assigned each candidate a distinct exploration strategy so the full
hypothesis space is covered more evenly across rounds. P42 extended it to
twelve — judgment by comparison between the two sequences, a single
most-salient local feature, a decision-rule mechanism, a running tally read
along the sequence, an exemplar/prototype account — so a six-candidate round
(three exploratory slots under the slot roles below) walks four rounds
without repeating a lens, and two experiments of two rounds cover the
battery exactly once. Only exploratory slots walk the battery: `_lens_offset`
advances by `max_iterations × exploratory_slots_per_round(candidate_count)`
per experiment, and `tests/golden/lens_schedule.json` was regenerated for
that schedule (a behaviour change, not a refactor).

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

### One retry and one repair per candidate slot (`pymc_orchestrator._Slot`)

In the 2026-09 sweep (`sweep_rerun`, 20 cells), 104 of 360 candidate slots
(29%) ended as "no candidate.py written" — most of them permission denials
(fixed in P34) — against 29 rejections of every other kind combined. The only
retry, the all-slots-empty round retry, requires every slot of a round to be
empty (~2% of three-slot rounds at that failure rate) and never fired. Of the
29 reasoned rejections, 17 were "predicts like existing model X"; the agent
never saw the reason and had no second attempt, and `CONTEXT.md` never told it
how to check that its model loads and samples before finishing.

P36 makes every slot end admitted or with a recorded reason the agent had a
chance to act on:

- A slot whose agent wrote no `candidate.py` (or whose agent process failed)
  is re-spawned once, in its own directory (`candidate_<i>_retry_1/`), with a
  fresh context, the same lens and a note saying the first attempt wrote
  nothing. A retry that stays empty is final.
- A candidate rejected at admission is re-spawned once
  (`candidate_<i>_repair_1/`) with the rejection reason injected verbatim into
  its prompt and the rejected files copied in as a starting point; its own
  rejected name is left out of "already tried — do not re-propose". A repair
  is always final, and a slot that did write a candidate never reads as an
  unfilled slot to the round guard — otherwise a repair that wrote nothing
  would trip the all-slots-empty retry and, with one slot, the systemic
  `AllCandidatesNoFileError`.
- Every attempt is a ledger line: retry and repair contexts carry
  ` retry 1` / ` repair 1` after `candidate <i> lens <j>`.
- `CONTEXT.md` documents one self-check command
  (`src/pipelines/inner_loop/check_candidate.py`, run with the pipeline's own
  interpreter) that runs the admission gates the agent can act on — import
  allowlist, loadable module-level model, finite logp, a short fit, finite
  ELPD-LOO — with `CANDIDATE_CHECK_DRAWS/TUNE/CHAINS = 100/100/1` from
  `mcmc_defaults.py`, no fit-cache writes, and no novelty check.

At most three agent attempts per slot. The consolidation plan's "do not bring
over candidate repair" exclusion is lifted by the loop-improvement plan §0.3;
the coupling concern above is answered by keeping the repair a prompt-only
mechanism — admission itself is unchanged and the novelty gate has no
exemption.

### The ledger stores hypotheses in full (`hypothesis_ledger.collapse_whitespace`)

Until P37, `_record` passed every hypothesis through `one_line(text,
limit=240)` before the JSONL line was written, so a hypothesis longer than 240
characters was cut with `…` *in the ledger itself* — the full text was gone,
not merely hidden — and the brief's markdown table could only ever show one
line per retired model. That is harmless while the ledger is a blacklist, and
load-bearing once candidate agents choose a refinement target from these
descriptions (P42).

P37 changes the write path and the rendering, and nothing else:

- `_record` stores the hypothesis with whitespace collapsed and no length
  limit (`collapse_whitespace`; `one_line` and its limit are gone).
- `render_markdown` renders one heading per retired model with the outcome
  detail and the full hypothesis as paragraphs of their own, instead of a
  table whose cells forced single lines and rewrote `|` characters.
- `LedgerEntry` gains no field: `from_json` requires an exact key-set match,
  so a new field would make every inherited ledger unreadable and
  `HypothesisLedger.create` would raise on the file experiment N-1 carried.
  A ledger written before P37 (its hypotheses ending in `…`) parses and
  renders unchanged.

### The incumbent record (`src/subjective_randomness/incumbent.py`)

Reading the three complete `motif_stack` cells of the September 2026 sweep
(`sweep_rerun` run2/run3/run5) showed the exported best model was
`local_representativeness` at every one of their 27 scoring steps: the loop's
output was its own best starting seed, refit on more data, and RMSE drift
(0.157 → 0.154) hid that. P38 makes "did the incumbent ever change, and was it
ever a discovered model" a first-class output so every later loop change is
judged on it:

- Every trajectory row (`trajectory.json`, `holdout.csv`) carries
  `incumbent_changed` (its `best_model` differs from the previous step's;
  never at global step 0) and `incumbent_is_discovered` (its `best_model` is
  not among the models scored at experiment 1's seed step — the project seeds
  the cell was seeded with, held-out GT excluded). Each cell's `trajectory.json`
  gains an `incumbent` block: the starting models, step count, number of
  changes, number of discovered-incumbent steps, the list of changes and the
  final incumbent. The offline re-analysis (`reevaluate_trajectories`) writes
  the same record.
- "Discovered" is read from the run record itself rather than from a manifest
  so archives and re-analyses need no configuration; the live harness
  cross-checks that starting set against the seed pool minus the held-out GT
  and raises on a stray name (it caught a test stub whose seed step named a
  superseded model).
- `scripts/subjective_randomness/incumbent_report.py` reports the record over
  a finished sweep from each cell's `agent_runs.tar.gz` or kept repo copy;
  `verify_holdout_run.sh` check 12 warns — never fails — on a cell with zero
  changes, because zero is the true baseline and must not block a run.

Validated against the archive: the three complete `motif_stack` cells report
0 incumbent changes and 0 discovered-incumbent steps over 27 steps. The two
`motif_stack` cells whose tasks failed (run1 after 4 steps, run4 after 5) did
each change incumbent once, to a discovered model; over all 20 cells the sweep
had 17 changes in 159 steps, and every complete `falk_konold_dp` cell, like
every complete `motif_stack` cell, never changed.

### No fallback critique battery; retry once, then no critique (`critique_round.py`)

In the September 2026 20-cell sweep (`sweep_rerun`) the critique agent never
produced a single test statistic: its first action was to read
`CRITIQUE_CONTEXT.md`, that read was denied as an external directory (P34),
and it exited after four log lines. The pipeline then silently wrote a
"default battery" — one statistic per varying numeric column excluding the
response and the row bookkeeping, which under the raw-only schema is exactly
one file, `fallback_mean_response.py`, the marginal choice rate any fitted
Bernoulli likelihood matches by construction. Every one of the six archived
rounds reported `0 of 1 test statistics show a significant discrepancy`, so a
dead subsystem looked alive through the sweep, a ceiling analysis and
`ANALYSIS_FINAL.md`. P35 inlines the context into the prompt (as the candidate
agent's documents already were), deletes the fallback, retries an agent that
wrote nothing once, and otherwise records the round as `no_critique` in
`history.json`; the verifier warns (not fails — a run is valid without a
critique) when no round of an experiment was critiqued.

### PSIS-LOO exact-trial exemption (`loo_reliability.py`)

Measured on the 2026-08/09 holdout sweeps, every non-finite k in the excluded
winners' fits was a constant-log-likelihood trial (40–720 per model, one per
participant × stimulus), and not one trial had a finite k above 0.7. Those
winners were dropped for nothing.

### Refinement slots: a depth mechanism (`model_zoo.slot_roles`, `candidate_agent._write_refinement_menu`)

In the three archived `motif_stack` cells of the September 2026 sweep the
exported best model was `local_representativeness` at every one of 27
scoring steps: the loop never beat its own starting seed. Reading the run
trees: 30 of 31 admitted models were pruned, and the loop's three mechanisms
— the novelty gate, the ledger's "already tried — do not re-propose", and
pruning — all push toward *new* mechanisms. Nothing let a partially correct
one be improved. Lens 0 said "refine one existing hypothesis" without naming
a target and forbade grafting; the brief forbade composition outright ("a
blended mega-model is not a hypothesis"); and the ledger forbade
re-proposing anything retired, so a promising loser such as
`bayesian_markov_alternative` (pruned at 415.5 nats, 15.7× dse) could not be
revived by design.

P42 gives each of a round's `candidate_count` slots a role:

- **Allocation.** With four or more slots: `C - 3` exploratory slots (the
  lens battery, as before), two slots that refine the incumbent, one slot
  that refines a non-incumbent model of the agent's choosing. Below four the
  refinement slots are given up one at a time — the second incumbent slot
  first, then the chosen slot, then the last incumbent slot — never the
  exploratory slot: `[explore]` at one, `[explore, incumbent]` at two,
  `[explore, incumbent, chosen]` at three. Exploratory slots come first in
  slot order so the lens walk is `lens_offset + iteration ×
  exploratory_per_round + exploratory_idx`.
- **The incumbent** named in the brief is the latest history step's
  `best_model` — the same ELPD-rank-among-reliable rule that selects the
  export and the critique incumbent — with its hypothesis, its `az.compare`
  standing and its source path.
- **The rules lifted, for refinement slots only.** The refinement briefs
  replace the one-hypothesis-no-blend clause with: the rule against grafting
  cues from other models and the rule against composing mechanisms are
  lifted; make one deliberate, stated change; change something that matters
  (the novelty gate still applies, unchanged). Exploratory briefs keep the
  clause, the lens, and the "do not re-propose" list. The shared theory
  prompt says the grafting rule is lifted in a refinement slot and that a
  many-heuristic blend is not a hypothesis in any slot.
- **The retired list becomes a menu** for refinement slots.
  `refinement_menu.md` lists the live non-incumbent models ranked by standing
  and the ledger's pruned models ranked by margin, each with its full
  hypothesis (P37), its standing or prune margin, and its source
  (`models/<name>.py`, or `models/pruned/<name>.py` when this run pruned it;
  a model pruned in an earlier experiment has no file in this tree and the
  menu says so). Rejected candidates never entered the set and are not
  targets. The margin is read back from the ledger `detail` that
  `_prune_losers` writes (`prune_margin_detail` / `parse_prune_margin`, one
  fixed format since the ledger's first commit) because `LedgerEntry` cannot
  gain a numeric field: `from_json` requires an exact key set, so a new field
  would make every inherited ledger unreadable.
- **No parsing.** The agent states in `hypothesis.md` which model it refined,
  in prose, because that is part of the claim. Nothing parses it or branches
  on it: no `refine_target.txt`, no regex, no ledger field, no novelty-gate
  exemption (P41's pool-based gate at 0.002 makes one unnecessary). The
  slot's *assignment* is what the ledger context records — `candidate 1
  refine incumbent <name>`, `candidate 2 refine chosen`, with ` retry 1` /
  ` repair 1` suffixes as before — and a retry or repair keeps its role.
  P48 reads which targets were chosen from the hypotheses.

Cost: a three-candidate round now has one exploratory slot instead of three,
so breadth per round drops until P45 raises `candidate_count` to six (three
exploratory, two incumbent, one chosen). That trade is the point: the sweep
measured breadth saturating with zero depth.

### The LOO design effect is measured, not assumed (`src/subjective_randomness/loo_design_effect.py`)

**Decision (P43, analysis only — no pruning code changed).** The inner loop
prunes on `elpd_diff > 2·dse` from a trial-level PSIS-LOO, which treats a
participant's response to a stimulus as an independent observation. A
holdout cell's 40 participants all answer the same 64 stimuli and the models
differ *by stimulus*, so the pointwise ELPD differences within a stimulus are
correlated and the trial-level `dse` is too small by a design effect. P43
measured it from the cached fits of the 20-cell September 2026 sweep (no new
MCMC) and wrote the recommendation for the user in
`$WORK_ROOT/LOO_DESIGN_EFFECT.md`.

**What was measured.** Every recorded scoring step's comparison (159 steps,
240 prune decisions, 167 prunes) rebuilt under three units: the loop's
trial-level PSIS-LOO — reproduced and checked against the recorded ELPDs, the
archived prune sets and the ledger margins, so the cache-to-decision mapping
is verified; a stimulus-clustered standard error of the same ELPD difference
(`sqrt(G · var(per-stimulus sums))`); and a leave-one-stimulus-out PSIS-LOO
with its own reliability verdict.

**Numbers.** The clustered `dse` is 1.99× the trial-level `dse` at the
median (1.05–5.54 over 207 comparisons; by ground truth 1.44 for
finite_experience_occurrence to 2.57 for motif_stack; larger on pooled data,
where the EIG-selected stimuli concentrate the disagreement). Under the
clustered unit 18 of the loop's 167 prunes (10.8 %) would have been kept —
all at 2–5× the trial `dse`, none at 5× or more, no survivor pruned, no
step's best model involved. Leave-one-stimulus-out PSIS-LOO gives the same
ratio (2.01) but degrades PSIS (20 of 381 fits unreliable against 6; one
model with 85 % high-k stimuli), changes the estimand, and disagrees with
the trial-level best at 8 of 159 steps (all ties).

**Recommendation (for the user; not applied).** Replace `dse` in
`_prune_losers`' rule with the stimulus-clustered standard error, keep the
2× multiplier, keep trial-level PSIS-LOO for the estimates, ranking and
export. Same estimand, no new MCMC, one prune in nine stays in the
uncertainty set — the near-miss models P42's refinement menu is meant for.
On human data the honest unit is a two-way cluster (stimulus × participant);
participants here are simulated i.i.d. given the stimulus.

**Two findings from running it.** The loop never prunes at a seed step (a
model carried into the next experiment can be far behind on the pooled data
and is still scored there; pruning runs only after a round), so the record
check compares prune sets at round steps only. And a stimulus whose forty
trials are all clipped has a grouped log-likelihood that is constant across
draws up to floating-point noise; arviz's PSIS returned NaN weights for that
near-constant column, so such groups are snapped to a constant before PSIS
(the exact-trial rule `loo_reliability` already applies).

### Models are fit concurrently (`pymc_inference.fit_models_cached`, `fit_models_to_cache`)

**Decision (P44).** `fit_models_cached` was a sequential `for` loop; each fit
ran four chains on four cores while the holdout task held eight CPUs, and
P45 multiplies the proposals per experiment by five. The models that need
MCMC are now sampled in a `ProcessPoolExecutor` of spawned workers: each
worker fits one model with `fit_model` (BLAS pinned to one thread through
`threadpoolctl`, a PyMC dependency, plus the thread-count environment
variables), persists its `.nc` into the existing content-addressed cache,
and the parent loads every fit from that cache — so a parallel fit is the
same `FittedModel`, with the same fingerprint, a sequential fit would have
produced (the acceptance test samples both ways and compares fingerprints
and ELPD-LOO). `fit_workers` caps the concurrency; the default is
`allocated CPUs // cores-per-widest-fit` from the scheduler affinity mask
(`allocated_cpus`), never below one, so `workers x chains` never exceeds
the allocation. It is not a sampler setting and takes no part in any cache
key. `holdout_recovery_array.sbatch` asks for 16 CPUs (four 4-chain fits at
once) and 64 GB (the September 2026 sweep peaked at 13.8–18.7 GB per task
fitting one model at a time under 32 GB).

**Where the batch is.** Cache hits made the change invisible on its own:
the experiment-start ELPD screen (`model_zoo._drop_nonfinite_elpd_models`)
fit every carried model one at a time, and by the time `compare_table`
called `fit_models_cached` everything was cached. The screen now samples
the whole set through `fit_models_to_cache`, the tolerant sibling that
reports failures by name instead of raising — a model whose fit fails is
dropped on that report and never fit a second time. The candidate real-fit
gate in `_admit_candidate_with_reason` still fits one candidate at a time,
by the admission order's design; it is the remaining sequential site.

**Two things the worker must do**, both found by the real-MCMC acceptance
test rather than the unit tests. A spawned child's default multiprocessing
start method is spawn, under which PyMC pickles the step method for its
chain processes — and the model lives in a module `load_pymc_model`
executed from a file, which a fresh interpreter cannot import ("The model
could not be unpickled"). The worker therefore sets the start method to
fork before sampling, as the sequential path forks from the main process.
And a worker's exception travels to the parent by pickle: PyMC's
`ParallelSamplingError` takes a chain number its pickled form does not
carry, so the parent could not rebuild it and the whole pool broke
(`BrokenProcessPool`) instead of one model failing. Every worker failure
is re-raised as `FitWorkerFailure("<Type>: <message>")`, with the original
traceback printed to the worker's stderr (the run log). A spawned child
also re-imports the entry script as `__mp_main__`, so every entry point
that reaches a fit needs the `if __name__ == "__main__":` guard; all of the
repo's have one.

**Measured.** Four project seed models on 1,280 real rows, 4 CPUs,
2 chains on 2 cores per fit: sequential 144 s, two workers 127 s (1.14×),
identical fingerprints and ELPD-LOO. The batch is dominated by one fit
(`motif_stack`, 107 s of the 144 s; the other three took 10, 9 and 18 s and
were entirely hidden behind it), so the parallel wall is the slowest fit
plus pool start-up. A batch's speedup is `sum / max` of its fit times,
capped at the worker count; on a batch of comparable fits it approaches
the worker count. CPU accounting (user+sys over wall) rose from 1.83 to
2.18 busy CPUs, and a pure-CPU probe confirmed the allocation runs four
processes at 3.98.

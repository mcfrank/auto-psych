# Loop-improvement plan, September 2026 — executable version

Phases **P34 … P48**, continuing the campaign whose phases P0–P33 are recorded
in `docs/consolidation_plan_2026_09.md`. That plan is history; **this file is
the plan in force.** Where the two disagree, this one wins, and §0.3 lists the
two places they deliberately disagree.

The driver, the work root, the clone and the progress markers are unchanged.

---

## 0. What this improves and why

### 0.1 The finding that motivates the whole plan

The 20-cell sweep (`$WORK_ROOT/sweep_rerun`) completed, and the recovery
ceiling job established that generation — not selection, pruning or finite
data — is the binding constraint. Reading the archived run trees turned up
something sharper. In all three archived `motif_stack` cells, across 27
scoring steps, the exported best model is `local_representativeness` **every
single time**:

```
run2  exp1/2/3, all iters -> local_representativeness, rmse 0.157 -> 0.157
run3  exp1/2/3, all iters -> local_representativeness, rmse 0.158 -> 0.154
run5  exp1/2/3, all iters -> local_representativeness, rmse 0.157 -> 0.154
```

`fitted_baseline.local_representativeness.rmse` is 0.155–0.158. So the loop's
output is its own best starting seed, refit on more data. 30 of 31 admitted
models were pruned; only `ideal_alternation_rate` ever survived, in one cell,
and it never became the incumbent. In run2 and run5 the `manifest_models`
carried into experiments 2 and 3 are identical to experiment 1's.

**The loop has never once beaten its own starting point.** That is the failure
this plan attacks, and "did the incumbent ever change to a discovered model"
is its primary metric (P38 makes it a recorded number; today it is 0 of 27).

### 0.2 The measured causes

Every number below comes from `$WORK_ROOT/sweep_rerun` and its `agent_runs.tar.gz`
archives; the phases that act on them re-derive them rather than trusting this
summary.

1. **Agents are denied write access to their own working directory.** Agent
   logs contain
   `permission requested: external_directory (<candidate_dir>/*); auto-rejecting`
   followed by `The user rejected permission to use this specific tool call.`
   The repo `opencode.json` grants `external_directory` only for `/tmp/**` and
   macOS temp paths. Across the sweep: **104 of 360 candidate slots** ended as
   `no candidate.py written` — 29% — against 29 rejections of every other kind
   combined. In `run3/motif_stack`, 16 of 18 candidate agents and **6 of 6**
   critique agents hit the rejection.

2. **The critique round has never produced a real critique.** The critique
   agent's first action is to read `CRITIQUE_CONTEXT.md` (its context is not
   inlined into its prompt, unlike the candidate agent's). That read is denied,
   so it writes zero test statistics and exits after 4 log lines. The pipeline
   then silently substitutes `_write_default_test_statistics`, which builds one
   statistic per *varying numeric column* excluding `participant_id`,
   `trial_index` and the response — and under the raw-only schema those
   exclusions cover every numeric column there is. All six archived rounds
   contain exactly one statistic file, `fallback_mean_response.py`: the
   marginal choice rate, which any fitted Bernoulli likelihood matches by
   construction. Hence `0 of 1 test statistics show a significant discrepancy`
   in every round. CriticAL has contributed zero signal to any run.

3. **A round is only retried when every slot fails.** `_is_all_no_file_round`
   requires all slots empty; with 3 slots at ~29% independent failure that is
   ~2% of rounds, and the retry did not fire once in the whole sweep. A round
   with 2 of 3 empty is accepted as-is.

4. **Rejected candidates get no feedback and no second attempt.** 17 of the 29
   non-permission rejections are `predicts like existing model X`. The agent
   never sees that message. `CONTEXT.md` also never tells the agent it may
   check that its model loads and samples before finishing, so it writes a PyMC
   model blind.

5. **The novelty gate is measured in the wrong place.** It compares posterior
   mean `p_left` on the 64 training stimuli at a 0.02 RMSE threshold. The 23
   archived rejection margins are bimodal — about 5 genuine re-skins clustered
   at ~0 (two predicting *identically*), and about 18 spread evenly from 0.006
   to the threshold, which is what distinct mechanisms that happen to agree on
   64 points look like:

   ```
   0.0000 x2  0.0001  0.0002  0.0004  0.0018  0.0031  0.0036  0.0061  0.0088
   0.0097 x2  0.0114  0.0115  0.0117  0.0130  0.0133  0.0152  0.0166  0.0169
   0.0173  0.0183  0.0191
   ```

   Removing the gate is not the answer: a model that predicts identically to
   the incumbent is *statistically tied* with it, so `elpd_diff ~ 0 < 2*dse`
   and pruning will never remove it — it survives every round, is carried
   forward, is refit on every scoring pass, and fills `existing_hypotheses.md`
   with near-identical entries.

6. **The loop has three breadth mechanisms and no depth mechanism.** The
   novelty gate, the ledger's "do not re-propose", and pruning all push toward
   new mechanisms; nothing lets a partially-correct one be improved. Lens 0
   says "refine one existing hypothesis" without naming a target and forbids
   grafting; the brief forbids composition outright; and the ledger forbids
   re-proposing anything retired. A promising loser like
   `bayesian_markov_alternative` (pruned at 415.5 nats, 15.7x dse) cannot be
   revived by design.

7. **Hypotheses are destroyed at ledger write time.** `one_line(text,
   limit=240)` truncates with `…` in `_record`, before the JSONL line is
   written, so the full text is gone — not merely hidden. This becomes
   load-bearing at P42, where the agent chooses a refinement target from these
   descriptions.

8. **opencode sqlite contention.** 28 `database is locked` events in the sweep
   logs with 3 concurrent agents. It is retried, but 6 candidates plus a
   critique agent (P45) roughly triples the contention.

### 0.3 Where this plan supersedes the consolidation plan

- `docs/consolidation_plan_2026_09.md` §2 says **"Do not bring over ...
  candidate repair"**. P36 deliberately reintroduces a bounded candidate
  repair (one attempt, rejection reason injected). That exclusion is lifted.
- That plan's §7 stop conditions still apply; §7 below adds to them.

### 0.4 What is explicitly out of scope

- `reserved_for_new` is hardcoded to `0.0` at `model_loop_runner.py:374` and
  `theory_weights()` drops it before the design sees it, so EIG only ever
  discriminates among in-set models. Changing the EIG objective is a research
  change, not a bug fix. **Do not touch it in this plan.**
- Swapping the candidate-agent model away from Gemini 3.1 Pro. The sweep in
  P46 must use the same agent backend and model as `sweep_rerun`
  (`opencode` / `google/gemini-3.1-pro-preview`) or the comparison is worthless.
  Fable 5.1 is the model executing *this plan*, not the model inside the loop.

---

## 1. Evidence base (read-only)

| Name | Path | What it is |
|---|---|---|
| `$SWEEP_RERUN` | `$WORK_ROOT/sweep_rerun` | The 20-cell sweep this plan is measured against. Per-cell `agent_runs.tar.gz`, `holdout.json`, `trajectory.json`, `mcmc_cache/`. |
| `$SWEEP1` | `$WORK_ROOT/sweep` | The first (crippled) sweep. Historical. |
| `$CEILING` | `$WORK_ROOT/ceiling` | Recovery-ceiling results: 35 cells, ceilings 0.004–0.012. |
| `$ANALYSIS_FINAL` | `$WORK_ROOT/analysis_final` | The offline re-analysis behind `ANALYSIS_FINAL.md`. |
| `$MOTIF_RECORDS` | `$SCRATCH/auto-psych/motif_stack_records` | Extracted readable records for the three archived `motif_stack` cells. |

These are **read-only**. Never write into them; copy anything you need into
`$WORK_ROOT` or the clone.

---

## 2. Where the agent works and what it may do

- **Clone:** `$REPO`, branch `consolidate/2026-09`. Work and commit here.
  Never push. Never touch `$SOURCE_REPO` (the user's checkout).
  The branch is behind `main`: **P34 merges `main` first** (the user merged the
  campaign plus two CI fixes — an `h5py<3.15` pin and three test fixes).
- **Progress dir:** `$WORK_ROOT/progress/` — phase markers, outside the repo.
- **Python:** `$VENV_PY`. You are on a Slurm compute node; running Python and
  the fast suite here is fine. Never write under `$HOME`.
- **Fast suite:**
  `$VENV_PY -m pytest -q -m "not slow" -p no:cacheprovider --continue-on-collection-errors -rf`
  Compare against `$WORK_ROOT/progress/baseline_failing_tests.txt` **by node
  ID, never by count.**
- **Static checks:** `git diff --check`; `$VENV_PY -m compileall -q src scripts`;
  `$VENV_PY -m pytest -q tests/test_python_sources_compile.py`.
- **Slurm:** you may `sbatch` only in **P39, P45, P46, P47** — nothing else.
  Never `scancel`, never `scontrol`, never poll or sleep-wait on a job. The
  driver waits by requeueing itself.
- **TDD, as the repo requires:** every behaviour change starts from a failing
  test. Run the relevant tests after each green step; the fast suite before
  each commit.
- **Fail loudly.** No silent defaults, no fallbacks, no `except: pass`.
- **Reading archives:** extract what you need from `agent_runs.tar.gz` into a
  scratch directory under `$WORK_ROOT`, never in place.

---

## 3. Phase contract (enforced by the driver)

Identical to `docs/consolidation_plan_2026_09.md` §3. In short: execute exactly
your phase; commit everything on the branch with a clean tree; write
`$WORK_ROOT/progress/P<k>.done` whose first line is `commit: <full sha of HEAD>`
followed by a short summary. On a stop condition (§7) write `P<k>.blocked`
instead and stop. A verdict phase may re-open an earlier phase by writing
`P<j>.retry<k>` and deleting `P<j>.done`, and then must **not** write its own
`.done`.

---

## 4. Phases

### P34 — Agent write access is unconditional

**Goal.** No agent session is ever denied access to its own working directory,
and a denial can never again pass as a quiet "no candidate.py written".

1. Merge `main` into the branch (`--no-ff`). Resolve conflicts; the fast-suite
   baseline must not gain node IDs.
2. **Diagnose empirically before fixing.** Extract one candidate agent log and
   one critique agent log from `$SWEEP_RERUN/run3/motif_stack/agent_runs.tar.gz`
   and reproduce the classification: determine *why* opencode treats a path
   under the agent's own cwd as an external directory. Record the finding in
   the done marker. Do not guess; the fix must follow the diagnosis.
3. Grant the run tree. The inner loop (or the holdout sbatch, whichever the
   diagnosis says is the right layer) must ensure the agent tree's
   `opencode.json` grants `external_directory` for the results root, the models
   directory and the responses directory. Existing grants are preserved.
4. **Loud failure.** After every agent session, scan its `agent.jsonl` for
   `auto-rejecting`. A permission denial is a systemic misconfiguration, not
   bad luck from one candidate: raise, do not record it as a per-slot
   rejection.
5. **Per-agent opencode store.** Give each spawned agent its own
   `XDG_DATA_HOME` (a subdirectory of its own working dir) so concurrent agents
   do not contend on opencode's shared sqlite database. Cite the 28
   `database is locked` events as the motivation in the commit message.

**Acceptance.** Unit tests for the config writing and the log scanning (a log
containing `auto-rejecting` must raise; a clean log must not). Fast suite at
baseline. The real end-to-end proof is P39 — do not claim it here.

### P35 — The critique round produces real statistics or none at all

**Goal.** CriticAL either works or is visibly absent. No more tautological
critiques.

1. **Inline the critique context** into the critique agent's prompt exactly as
   `_build_candidate_prompt` inlines `CONTEXT.md`, `CANDIDATE_BRIEF.md` and the
   rest. `CRITIQUE_CONTEXT.md` still lands on disk for audit, but the agent
   never needs to read it.
2. **Delete the fallback battery.** Remove `_write_default_test_statistics` and
   its call site entirely, along with its tests. This is a silent fallback of
   exactly the kind `CLAUDE.md` forbids, and it made a dead subsystem look
   alive through a full sweep, a ceiling analysis and an `ANALYSIS_FINAL.md`.
3. **Retry, then skip loudly.** Zero statistics from the critique agent ⇒ retry
   the agent once. Still zero ⇒ write no `critiques.md`, log loudly, and record
   the round as having no critique. `_run_critique_round` already returns
   `None` and the candidate round already proceeds without a critique; use
   that path.
4. **Make absence visible.** Record per-round critique status (statistics
   proposed, significant, or "no critique") in the run record, and have
   `scripts/subjective_randomness/slurm/verify_holdout_run.sh` flag a finished
   run in which no round produced a critique.

**Acceptance.** Tests for the retry-then-skip path and for the recorded status.
`grep -r "default battery\|_write_default_test_statistics" src scripts` returns
nothing. Fast suite at baseline.

### P36 — No candidate slot is lost silently

**Goal.** Every slot either produces an admitted model or a recorded reason the
agent had a chance to act on.

1. **Per-slot retry.** A slot that wrote no `candidate.py` is retried once, on
   its own, in its own directory. Keep the existing all-slots-empty round retry
   as the outer guard; the per-slot retry is what actually fires.
2. **One repair attempt per rejected candidate.** When `_admit_candidate`
   rejects, re-spawn that slot's agent once with the rejection reason injected
   verbatim into the prompt (`predicts like existing model X`, `not a loadable
   PyMC model: ...`, `non-finite ELPD-LOO`, ...). One attempt only; a second
   rejection is final. This lifts the consolidation plan's "do not bring over
   candidate repair" (§0.3). Every attempt and outcome goes to the ledger.
3. **Let the agent test its own model.** `CONTEXT.md` gains a documented
   command the agent can run to check that `candidate.py` loads as a
   module-level `model: pm.Model` and completes a short fit, with the exact
   interpreter path and a small draws/tune setting. Keep it cheap — this must
   not become a full production fit.

**Acceptance.** Tests: an empty slot is retried exactly once; a rejected
candidate is repaired exactly once and the second rejection is final; both
attempts appear in the ledger. Fast suite at baseline.

### P37 — The ledger stores hypotheses in full

**Goal.** A hypothesis written by an agent is never truncated by the pipeline.

1. `_record` stores the hypothesis with whitespace collapsed but **no length
   limit**. `one_line`'s truncation is removed from the write path.
2. `render_markdown` renders full hypotheses. The markdown table forces
   single-line cells, so switch to a heading-or-definition-list layout in which
   a multi-sentence hypothesis survives intact.
3. Do **not** add fields to `LedgerEntry`. `from_json` requires an exact
   key-set match, so any new field makes every inherited ledger unreadable and
   `HypothesisLedger.create` raises on the file experiment N-1 carried.

**Acceptance.** A 2000-character hypothesis round-trips through append →
`entries()` → `render_markdown` with no `…` and no loss. An inherited ledger
written before this change still parses. Fast suite at baseline.

### P38 — Record whether the incumbent ever changes

**Goal.** Make §0.1's number a first-class output, so every later change is
judged on it rather than on RMSE drift.

1. The run record gains, per scoring step: the exported best model, whether it
   changed from the previous step, and whether it is a discovered model (not a
   project seed). Aggregate per cell: number of incumbent changes, number of
   steps at which a discovered model was the incumbent.
2. `verify_holdout_run.sh` flags a finished run with zero incumbent changes.
   This is a **warning, not a failure** — zero is the current true value and
   must not block a run.
3. **Validate against the archive.** Run the new reporting over the three
   archived `motif_stack` cells in `$SWEEP_RERUN` and confirm it reports 0
   incumbent changes across 27 steps. A different answer means the metric is
   wrong; fix the metric, not the expectation.

**Acceptance.** Tests plus the archive validation, with the numbers in the done
marker. Fast suite at baseline.

### P39 — Submit the Phase-A smoke cell

**Goal.** Prove P34–P38 end to end on a real cell before building on them.

Submit **one** holdout cell: ground truth `motif_stack`, the same agent backend
and model as `sweep_rerun` (`opencode` / `google/gemini-3.1-pro-preview`), smoke
settings (2 experiments, 1 inner-loop iteration, 3 candidates), seed 101, under
`$WORK_ROOT/phase_a_smoke`. Keep the repo copy (`KEEP_REPO_COPY=1`) so the
agent logs can be read directly.

Write `$WORK_ROOT/progress/phase_a_smoke_jobs.json` with label `smoke` and the
submitted job id. Submit nothing else.

### P40 — Phase-A verdict

**Goal.** Decide whether the fixes worked, on evidence, before Phase B.

Read the smoke run and report, in `$WORK_ROOT/PHASE_A_VERDICT.md`:

| Criterion | Pass condition | Baseline (`sweep_rerun`) |
|---|---|---|
| Permission denials | **zero** occurrences of `auto-rejecting` in any `agent.jsonl` | 16/18 candidate, 6/6 critique agents |
| Slot fill rate | every slot wrote `candidate.py`, or its retry did | 70% (256/360) |
| Critique statistics | every round has ≥ 2 agent-written statistics, or a recorded "no critique" | 1 pipeline-written statistic per round |
| `database is locked` | zero | 28 across the sweep |
| Admission rate | reported, not gated | 62% |
| Incumbent changes | reported, not gated | 0 of 27 |

The first four are pass/fail. If any fails, re-open the owning phase
(`P<j>.retry<k>`, delete its `.done`, no `.done` of your own) rather than
proceeding.

### P41 — A novelty gate that measures novelty

**Goal.** Reject re-skins; stop rejecting distinct mechanisms that happen to
agree on 64 stimuli.

1. **Measure on a broad pool.** `_min_prediction_rmse` compares posterior-mean
   `p_left` on an independently generated stimulus pool rather than the
   training stimuli. Generate it inside the loop from its own seed. **Do not
   reuse the recovery harness's eval pool** — the loop must not select models
   on the stimuli it is later scored against; state this in the code comment.
2. **Recalibrate.** Default `novelty_rmse_threshold` becomes `0.002`, with the
   §0.2(5) distribution cited in the decision record as the calibration.
3. **Make it a knob.** Plumb `novelty_rmse_threshold` through the holdout
   config → `scripts/subjective_randomness/holdout_recovery.py` →
   `src/subjective_randomness/holdout_recovery.py` → `model_loop_runner`. It is
   currently reachable only from `src/pipelines/outer_loop/run.py`, so the
   holdout config cannot set it and it cannot be A/B'd.

**Acceptance.** Tests: the pool is loop-generated and distinct from the eval
pool; a model predicting identically to an existing one is still rejected; a
model differing only off the training stimuli is admitted; the config knob
reaches the inner loop. Fast suite at baseline.

### P42 — Give the loop a depth mechanism

**Goal.** Let a partially-correct mechanism be improved across rounds instead
of dying once, without giving up the breadth that novelty search provides.

1. **Lens allocation.** Per round, with `candidate_count = C`:
   - 2 slots: **refine the incumbent**, named explicitly in the brief;
   - 1 slot: **refine a non-incumbent model of the agent's choosing**;
   - the remaining `C - 3` slots: exploratory lenses, as today.

   Define the degradation for `C < 4` explicitly and test it (at `C = 3`: one
   incumbent-refinement, one agent-chosen, one exploratory).
2. **Suspend the rules that forbid refinement, for refinement slots only.** The
   anti-grafting clause in lens 0 and the "a blended mega-model is not a
   hypothesis" clause in the brief do not apply to those slots. Exploratory
   slots keep both.
3. **The retired list becomes a menu for those slots.** For the agent-chosen
   slot, render the available targets — live non-incumbent models *and* pruned
   models — as one ranked list with full hypotheses (P37 makes this possible),
   standing or prune margin, and the source path (`models/<name>.py` or
   `models/pruned/<name>.py`). Frame it as a menu, not a blacklist. Exploratory
   slots keep the existing "do not re-propose" framing.
4. **No parsing.** The agent states in `hypothesis.md` which model it is
   refining, in prose, because that is part of stating the claim. **Nothing in
   the pipeline parses it or branches on it.** There is no `refine_target.txt`,
   no regex, no ledger field, no novelty-gate exemption — P41's recalibrated
   gate makes an exemption unnecessary, and the "do not re-propose" scoping is
   a per-slot prompt-template difference, since the ledger is only ever
   appended to and rendered (`candidate_agent.py:274` is its single read).
5. **Extend the lens battery** from 7 to at least 12 exploratory lenses, so a
   6-candidate round at P45 never repeats a lens within a round.

**Acceptance.** Tests for the allocation at several `C`, for the per-slot brief
differences, and for the menu rendering including pruned models with full
hypotheses. Fast suite at baseline.

### P43 — Measure the LOO design effect (analysis only, no loop change)

**Goal.** Decide the pruning unit from a number instead of an intuition.

With 40 participants over the same 64 stimuli, trial-level LOO treats 2560
correlated rows as independent. Using the cached fits in
`$SWEEP_RERUN/run*/*/mcmc_cache` (**no new MCMC**), compute for each archived
cell both trial-level and stimulus-grouped `elpd_diff` and `dse`, and the ratio
between the two `dse` values.

Write `$WORK_ROOT/LOO_DESIGN_EFFECT.md`: the measured ratios, how many archived
prune decisions would flip at the current `2*dse` threshold under each unit,
and a recommendation. **Change no pruning code in this phase** — the
recommendation is for the user.

**Acceptance.** The memo exists with per-cell numbers and a clear
recommendation. Any script it needs is committed and tested.

### P44 — Fit models in parallel

**Goal.** Buy the compute headroom P45 needs.

`fit_models_cached` is a sequential `for` loop; each fit uses 4 chains on 4
cores while the task holds 8 CPUs.

1. Fit models concurrently with a `ProcessPoolExecutor`; children write their
   `.nc` into the existing on-disk cache and the parent loads from it. Set
   `OMP_NUM_THREADS=1` in the children so BLAS does not oversubscribe.
2. Concurrency is a parameter, defaulting so that `workers * chains` does not
   exceed the allocated CPUs.
3. Raise `--cpus-per-task` in
   `scripts/subjective_randomness/slurm/holdout_recovery_array.sbatch` to match
   (16 on `normal`; note in a comment that `-p hns` allows up to 256 cores,
   26 GB/core and 7 days if a future sweep needs it). `normal` caps at 8 GB/core.

**Acceptance.** An equivalence test: the same models fit sequentially and in
parallel produce identical cache fingerprints and identical ELPD-LOO values.
Fast suite at baseline. Report a measured speedup on a real multi-model fit in
the done marker.

### P45 — Scale the inner loop, and submit the scale smoke

**Goal.** More proposals per experiment, proven to run within walltime.

1. `max_iterations: 2 -> 5`, `candidate_count: 3 -> 6`,
   `n_critique_proposals: 8` (already the default) in
   `scripts/subjective_randomness/configs/holdout_recovery.yaml`. The lens
   battery must already be ≥ 12 (P42).
2. Confirm agent parallelism covers 6 concurrent candidates and that the
   critique agent is spawned alongside rather than serially blocking the round.
3. Submit **one** scale smoke cell: `motif_stack`, gemini, full scaled settings,
   seed 101, under `$WORK_ROOT/scale_smoke`, `KEEP_REPO_COPY=1`. Write
   `$WORK_ROOT/progress/scale_smoke_jobs.json` with label `smoke`.

Archived cells took 1.5–6.5 h at the old settings against a 24 h wall; 5x the
slots must fit within the wall *with* P44's parallelism. That is what this
smoke measures.

### P46 — Scale verdict, then launch the recovery sweep

**Goal.** Only launch 20 cells if one cell works.

1. Read the scale smoke. Write `$WORK_ROOT/SCALE_VERDICT.md`: wall-clock,
   peak memory, slot fill rate, admission rate, critique statistics per round,
   incumbent changes, and whether 5x30 slots fit the 24 h wall with margin. If
   it does not fit, re-open P44 or P45 rather than launching.
2. If it passes, launch the sweep: 5 repeats x 4 ground truths = 20 cells, base
   seed 100, **`opencode` / `google/gemini-3.1-pro-preview`** (§0.4), under
   `$WORK_ROOT/sweep3`. Write `$WORK_ROOT/progress/sweep3_jobs.json` with label
   `raw`.

### P47 — Submit the evaluation

Wait for `sweep3_jobs.json`, then submit the RMSE evaluation over
`$WORK_ROOT/sweep3` on the same exhaustive eval pool used for `sweep_rerun`, so
the two are directly comparable. Write
`$WORK_ROOT/progress/analysis3_jobs.json` with label `analysis`.

### P48 — Results

Write `$WORK_ROOT/RESULTS_LOOP_IMPROVEMENT.md`:

1. **Primary:** incumbent changes per cell, and how often a discovered model
   was the incumbent, against the baseline of 0 of 27.
2. **Secondary:** RMSE per cell and per ground truth, matched-seed paired
   deltas against `sweep_rerun`, against the §8 margin, and the distance to the
   `$CEILING` numbers (0.004–0.012).
3. **Process:** slot fill rate, admission rate, permission denials, critique
   statistics per round, refinement slots and which targets were chosen (read
   from the hypotheses — nothing records it structurally, by design).
4. **Honest verdict:** state plainly whether the loop beat its starting point,
   and if it did not, which of §0.2's causes remain.

---

## 5. Test-baseline discipline

`$WORK_ROOT/progress/baseline_failing_tests.txt` is the reference. After every
phase the failing set must be a subset of it, compared **by node ID**. A new
failure caused by the phase is fixed in that phase. A new environment
collection error is recorded in the done marker. Nothing else is acceptable.

## 6. Commit discipline

One commit per numbered behavioural step where practical; merge commits with
`--no-ff`; messages say what and why and name the phase (`[P36]`). Never amend
a merge. Never rewrite history. Never push.

## 7. Stop conditions (write `P<k>.blocked`)

In addition to `docs/consolidation_plan_2026_09.md` §7:

- P34's diagnosis cannot establish *why* opencode classifies the run tree as
  external — fixing it by trial and error is not acceptable;
- a phase would have to weaken or delete a test to make the fast suite pass;
- P39 or P45's smoke cannot be submitted without changing more than a knob;
- P40's pass/fail criteria are not met after one re-opening of the owning phase;
- P46 would launch the sweep although `SCALE_VERDICT.md` does not pass;
- the sweep in P46 would run with any candidate-agent model other than
  `google/gemini-3.1-pro-preview` (§0.4);
- a change would touch `reserved_for_new` or the EIG objective (§0.4);
- a change would touch the ground-truth registry models, the family twins, or
  the held-out parameters;
- anything requiring a write to `$SOURCE_REPO`, a `git push`, or a `scancel`.

## 8. Metric protocol

Unchanged from `docs/consolidation_plan_2026_09.md` §8: RMSE primary, KL regret
secondary, Pearson r / bias / calibration descriptive; the repeat is the
stochastic unit; matched-seed paired deltas; predeclared practical margin
**0.02 RMSE**.

Added here, and **primary for this plan**: the **incumbent-change count** — how
many scoring steps exported a different model than the previous step, and at
how many steps a discovered (non-seed) model was the incumbent. Baseline: 0 and
0, over 27 steps in the three archived `motif_stack` cells.

## 9. Comparison baseline

`$SWEEP_RERUN` (`$WORK_ROOT/sweep_rerun`), matched by seed: repeats at base
seed 100, ground truths `falk_konold_dp`, `motif_stack`,
`finite_experience_occurrence`, `local_representativeness`. Its process
statistics are in §0.2 and its ceiling in `$CEILING`.

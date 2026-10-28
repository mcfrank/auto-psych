# Outer loop — `src/pipelines/outer_loop/` (`run.py` → `orchestrator.py`)

Moved verbatim from `CLAUDE.md` (2026-10-28); `CLAUDE.md` keeps the rules that span files.


`AGENT_KEYS = ["2_design", "3_implement", "4_collect", "5_model_loop"]` (there
is no stage 1). First the model set: experiment 1 seeds from
`projects/<id>/seed_models/`, later experiments carry forward the previous
experiment's set — only once that experiment's model-loop stage validates. Seed
and carry are idempotent and atomic (built beside `cognitive_models/` and
renamed in, `src/runtime/atomic_files.py`), which makes `--resume` and
`--agent <stage>` safe.

- `2_design` — **programmatic.** `eig.design_exhaustive` scores the task's whole
  design pool and greedily selects the max-joint-EIG set (64 by default) →
  `design/stimuli.json`. The EIG counts every participant's response
  (`n_responses` = the participant count, required everywhere). Selection stops
  once the best gain is within Monte Carlo noise of zero; the remaining slots
  are filled by single-response EIG (`source: eig_single_response_fill`).
  Experiment 1 uses prior predictions; later experiments use each model's
  posterior on all data so far, **loaded from the previous model loop's own
  fits** (`load_cached_fit`, at the run's sampler settings): nothing is
  sampled, and a missing fit raises `FitNotCached` (user decision 2026-10-08;
  `DESIGN_TWIN_*` survive only for `validate_lazy_eig.py`).
  The search is lazy batched greedy in float32 (an approximation validated by
  `scripts/subjective_randomness/validate_lazy_eig.py`), and each Monte Carlo
  scenario **leaves out the draw that generated it** (`DESIGN_LEAVE_ONE_OUT`),
  so a model needs ≥ 2 draws.
- `3_implement` — the one coding-agent stage: writes a jsPsych experiment
  (skipped in `simulated_participants_nobrowser`); a Firebase/Prolific deploy
  follows when `--deploy-target != none`. The page builds its trials from
  `jsPsych.randomization.shuffle(STIMULI)` (a fresh order per participant; the
  validator rejects the embedded order) and counterbalances side per trial.
- `4_collect` — programmatic: writes `data/responses.csv`.
- `5_model_loop` — runs the inner loop, exports the live set, writes
  `model_registry.yaml` and last `model_loop/export_complete.json`, which the
  validator checks against `cognitive_models/`, its ledger and the registry.
  Every (re)start goes through `begin_model_loop_stage`: the first records the
  input set (`cognitive_models_input/`) and the run's notes
  (`agent_notes_at_start/`); a restart restores both — notes from the abandoned
  attempt are **discarded** (later agents read the run tree) — and empties
  `model_loop/`. A `model_loop/` with content but no recorded input raises.

Coding stages that fail their validator (`_validate_*` in
`orchestrator_validators.py`) are re-spawned with the error as feedback;
programmatic stages fail terminally.


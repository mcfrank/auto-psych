# Supporting modules, projects, the number game, ablations, results

Moved verbatim from `CLAUDE.md` (2026-10-28); `CLAUDE.md` keeps the rules that span files.

## Supporting modules and rules

- **Agent isolation** (`src/runtime/agent_sandbox.py`): agents run in
  bubblewrap (jobs stop without `bwrap`), their tree read-only, only their own
  directories writable, with an allowlisted environment (the backend's own
  login, no other secrets, no `SLURM_*`). Agent trees hold only `src/` and the
  run tree (`agent_tree.exclude`). `scan_gt_name.sh --before-agents` stops a
  cell whose tree names its held-out ground truth — **never write a
  ground-truth model name into anything under `src/` that agents see.**
- **Raw-only data.** `responses.csv` carries only `sequence_a, sequence_b,
  participant_id, trial_index, chose_left` (`outer_loop/columns.py`); every
  model computes its own features (`compute_features` or `prepare_observed`
  hook). Full collected rows (with Prolific IDs) go to
  `<project>/raw_collected/`, which no agent gets; `run.py` refuses a
  collecting run whose output tree is inside the repo (set
  `AUTO_PSYCH_OUTPUT_DIR`). `tests/test_agent_prompts_raw_only.py` fails if an
  agent-facing text presents a feature column as data.
- **Design screening is recorded** in `design/screened_out.json`: models that
  cannot be predicted on the pool (bookkeeping columns, their own errors,
  undefined predictions) are left out of that design on record. Person-level
  models are predicted as 40 new participants (user decision 2026-09-30;
  `docs/person_level_models.md`). Missing stimulus columns raise
  (`MissingStimulusColumns`).
- `src/models/mcmc_defaults.py` is the **single source of sampler defaults**.
- `src/models/pymc_inference.py`: fits cached on (model sha, csv sha, sampler
  settings) in-process and as `<name>.<fingerprint>.nc`; concurrent fits in a
  process pool sized to the allocated CPUs; a wave's candidates prefit in
  time-limited children before sequential admission. **An infrastructure
  failure is never a model's** (`INFRASTRUCTURE_ERRORS`,
  `FitInfrastructureFailure`): it raises, never a drop or a rejection. Each
  fit process gets its own cache and PyTensor compile directory.
- `src/model_comparison/`: ELPD-LOO softmax posterior (overconfident, a report
  field), the `az.compare` table, grouped CV (`cv.py`).
- `src/critique/ppc.py`: agent-written test statistics scored against
  posterior-predictive replicates (two-sided p + BH-FDR q). The critique runs
  **before** the round's candidates (its `critiques.md` goes into every brief);
  no usable statistic after one retry → the round runs uncritiqued and records
  `"no_critique"`. There is no fallback battery.
- `src/runtime/coding_agent.py`: `run_coding_agent` is the single launcher;
  backend = explicit arg → `CODING_AGENT` → `opencode`.
- `src/subjective_randomness/incumbent.py`: the **incumbent record** (did the
  exported best model change, is it discovered), the primary metric for
  improving the loop; `incumbent_report.py` summarises a sweep.
- Sweep summaries (`sweep_cells.py`, `reporting.aggregate_holdout_trajectories`)
  list partial and missing cells, pool by position within an experiment, and
  average the loop and every baseline over the same cells.

## Projects vs. the research library

- `src/pipelines/outer_loop/projects/<id>/`: **assets the pipeline consumes**
  (`problem_definition.md`, `task_description.md`, `seed_models/`, a gitignored
  per-run `prolific_config.yaml`). Adding a project = adding an asset dir.
- `src/subjective_randomness/`: a **harness-only research library** (agent
  trees exclude it): the PyMC model families, pair pools, recovery harness.
  `pymc_model_families/` is the frozen recovery registry, and the project's
  `seed_models/` mirrors it **byte for byte** (a test checks;
  `sync_seed_models.py` copies). Editing a seed or registry file changes its
  hash, hence every cached fit, and what agents read. Ground-truth generating
  parameters live in the recovery configs' `seed_params` (impossible models'
  explicitly in `gt_models`).
- `src/tasks/`: the `TaskSpec` per project (design pool, stimulus rows,
  prediction variable, lens battery) that keeps the pipeline generic.

## The number game

`projects/number_game/`, `src/tasks/number_game.py`; three seeds (Tenenbaum
2000 and two ports of Bigelow's numpyro models); the dataset
(`data/number-game/`, gitignored) is read by `src/number_game/`. Recovery:
`scripts/number_game/slurm/submit_recovery.sh`; `data_source: dataset` runs
draw responses from the dataset instead (`submit_dataset_run.sh`). These runs
use `fit.method: warm_start_nuts`, grouped-CV selection (`inner_loop.selection`)
and `hide_participant_id`.

## Ablating the sr-oct26 finals

`src/subjective_randomness/ablation/` (`docs/sr_ablation.md`): each live run's
final rewritten with switchable mechanisms (`finals/run{1,2,3}.py`; all on is
the final, tested), and backward elimination by 5-fold CV grouped by stimulus
pair on all runs pooled. Slurm plan/worker arrays:
`scripts/subjective_randomness/slurm/submit_ablation.sh`.

## Ablating the number-game final

`src/number_game/ablation/` (`docs/ng_ablation.md`): the dataset run's final
(`even_rule_digit_size_principle_counting`) rewritten with its 13 logit terms switchable
(`final.py`), and the same backward elimination on the run's own 7,560 responses, by 5-fold CV
grouped by example set (`number_set`) with SEs clustered by example set. The rule, the variants
and plan/worker are shared with the sr-oct26 ablation in `src/ablation/` (each domain's
`eliminate.py` names its finals, CV unit and clusters); a change there changes both. Slurm:
`scripts/number_game/slurm/submit_ablation.sh` (each worker array task fits 5 (model, fold)
tasks; one fit is ~6 min).

## Inspecting results

`src/viewer/` (`python -m src.viewer.server`; `freeze.py` makes a static site);
`src/monitor/` is the live dashboard for an in-progress human study (first job:
catching participants who answer one side every trial).


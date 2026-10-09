# Handoff: archive run 2, promote the live seeds, design experiment 1

For the local session that runs RSA jobs on Sherlock. Three compute-only jobs;
no agents and no participants. Decisions: `PLAN.md`, "Decisions
(2026-10-08, the live campaign: claim 2)".

## 0. Pull

```bash
cd ~/auto-psych && git fetch origin && git checkout auto-rsa && git merge --ff-only origin/auto-rsa
export WORK_ROOT=$SCRATCH/auto-psych/rsa_run2   # run 2's sweep (the scripts' default)
mkdir -p "$WORK_ROOT/logs"
```

`git fetch origin auto-rsa` leaves `origin/auto-rsa` stale on git 1.8: fetch the remote, then merge.

## 1. Archive run 2 (any time; before $SCRATCH's 90-day purge)

```bash
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/archive_sweep.sbatch
```

- **Where it goes:** `$OAK/auto-psych/archive/rsa_run2` if Oak is mounted, else `$GROUP_HOME/auto-psych/archive/rsa_run2` (`ARCHIVE_ROOT` overrides).
- **What it contains:**
  - the prepared data and recovery simulations;
  - the cell records and Slurm logs;
  - each loop directory, with the agents' logs and opencode sessions;
  - the fit caches (`ARCHIVE_FITS=0` skips them if space is short).
- **Before writing anything** it stops if any file it would archive holds the agents' API key. It also rewrites each cell's `agent_activity.md`, listing web tool calls apart from URLs that only appear in log text.
- **Bring back:**
  - copy `SHA256SUMS` and `MANIFEST.json` into `data/rsa/sherlock_run2/archive/`;
  - add one line with the archive's path to `data/rsa/sherlock_run2/README.md`;
  - copy each `$CELLS_ROOT/<cell>/agent_activity.md` over `data/rsa/sherlock_run2/<cell>/agent_activity.md`.
- **Report:** the archive's size.

## 2. Promote the seeds (about 1-3 h)

```bash
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/promote.sbatch
```

- **What it does:**
  - takes the 143 models the three real cells admitted;
  - refits them on train + test (`$WORK_ROOT/promote_work/all_trials.csv`), 20 fits at once;
  - groups them into 10 by their predictions;
  - keeps the best of each group by grouped CV, then adds `rsa_l2`.
- **Writes:** `data/rsa/live_seeds/models/` and `promotion.json`. Commit both: model files and aggregates only.
- **Report:**
  - the 11 promoted names;
  - any `fit_error` or unconverged candidates in `promotion.json`;
  - the wall time.

## 3. Design experiment 1 and its power (about 1 h, after step 2)

If the first `design.sbatch` (submitted with the old 60-display, 64-trial settings) ran, discard its
`design_n100/`: the PI set 12 trials per participant on 2026-10-08. Pull, then:

```bash
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/design.sbatch
```

- **What it does:**
  - a participant answers 10 designed displays (plus 2 catch trials), a balanced subset of the design;
  - for designs of 10, 20 and 30 displays (`DISPLAYS`), it picks the displays by joint EIG for 200 participants (`PARTICIPANTS`);
  - it scores each design at 100, 200 and 300 participants (`POWER_PARTICIPANTS`): how often the model that generated simulated data ends with the highest posterior.
- **Reuses:** step 2's fits; only `rsa_l2` is fitted.
- **Writes:** `data/rsa/live_seeds/design_t10/design_d{10,20,30}.json` and `eig.json`. Commit them.
- **Report:** the log's power lines (one block per design size).

The power tables set the design size and the number of participants per experiment (PI decision pending).

## 4. Simulated dress rehearsal: cumulative vs live-only selection (after step 2)

PI 2026-10-09: try both ways the live loop could select. Agents are on credits; no participants.

JAX now runs on one CPU in every process of every RSA job (`_env.sh`'s `XLA_FLAGS`, Research Computing 2026-10-09; the fit workers already did), agents' self-checks included.

```bash
cd ~/auto-psych && git fetch origin && git merge --ff-only origin/auto-rsa   # after step 2's commit
mkdir -p "$SCRATCH/auto-psych/rsa_rehearsal/logs"
sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_rehearsal/logs/%x_%A_%a.out" \
  scripts/rsa/slurm/outer_rehearsal.sbatch
```

- **What it runs:** two array tasks, each one run of 3 experiments of 100 simulated people (PI 2026-10-09; the live campaign's size) x 10 of 20 displays, with real Gemini agents (sandboxed, no network):
  - task 0 selects on all data so far (cumulative);
  - task 1 fits on all data but selects on the live trials only.
- **Pairing:** both use the same seeds, the same hidden ground truth and the same seed, so experiment 1 is identical in the two; they differ only in how the inner loop selects.
- **The ground truth:** picked by rule (`src.rsa.outer.ground_truth`: the starting model farthest from every promoted seed) into `$SCRATCH/auto-psych/rsa_rehearsal/ground_truth.json`, and withheld from the agents' tree.
- **Its own staged code** in `$SCRATCH/auto-psych/rsa_rehearsal` (the run-2 staging is older code).
- **Time:** each experiment is a design (~15 min) and an inner loop of up to 5 rounds (several hours on ~52k trials): expect 15-30 h per task; the limit is 48 h. Resubmitting a task resumes it.
- **Bring back** (into `data/rsa/rehearsal/<cell>/`, for `rehearsal_all` and `rehearsal_live`):
  - `$CELLS_ROOT/<cell>/outputs/` without the trial-level CSVs (`experiment*/data/*.csv`, `model_loop/responses.csv`, `.cv/fold_*.csv`): the designs, `participants.json`, and each inner loop's history, ledger, models and export;
  - `$CELLS_ROOT/<cell>/private/` (the configuration, `experiment*/prospective.json`, `experiment*/recovery.json`; no fit cache);
  - `ground_truth.json` once, at `data/rsa/rehearsal/`.

  The simulated responses are not real people's data, but they are generated from the real existing data's fit: keep them on Sherlock like other trial-level files.
- **Report per task and experiment:**
  - the exported model and its RMSE to the ground truth (`recovery.json`);
  - the closest live model before and after;
  - the prospective `live_vs` lines;
  - the wall time and any failure.

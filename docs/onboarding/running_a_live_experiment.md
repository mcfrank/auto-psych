# Running a live experiment (Prolific + Firebase) on Sherlock

Every command here was checked against the scripts and command-line
interfaces on 28 September 2026. **None was run against Prolific or
Firebase**; those steps are described from the code. Placeholders: `$REPO` is
your checkout, `$WORK_ROOT` is where live runs write (default
`$SCRATCH/auto-psych/outer_loop_live`), `<label>` is a run label you choose.

## 1. Accounts and credentials

You need a Sherlock account, a funded Prolific researcher account, access to
the Firebase project `auto-psych-2c5da`, a login for the coding agents, and
IRB approval for the exact text in `templates/consent.txt` (shown verbatim as
the first page).

Credentials go in `$REPO/.secrets`, one `KEY=value` per line (gitignored;
never commit, copy or print it; `chmod 600`). `_env.sh` exports every key into
the job.

| key | used for |
|---|---|
| `PROLIFIC_API_TOKEN` | creating, publishing, polling and pausing studies |
| `FIREBASE_TOKEN` | `firebase deploy` from a compute node (make it with `firebase login:ci` on a machine with a browser) |
| `AUTO_PSYCH_RESULTS_TOKEN` | shared secret protecting `/results`; make one with `openssl rand -hex 32` and keep it the same for every deploy and collection |
| `GOOGLE_API_KEY` | the Gemini login of the default agents (opencode), unless opencode is logged in some other way on your account (not checked) |

With `coding_agent: claude`, state how the agents are billed with
`claude_auth` in the config: `subscription` needs `CLAUDE_CODE_OAUTH_TOKEN`
(from `claude setup-token`), `api` needs `ANTHROPIC_API_KEY`. There is no
default; the run stops before any agent starts without the mode or its key,
and each agent gets only that one credential. The agents never see the
Prolific, Firebase or results tokens.

## 2. One-time setup

```bash
export REPO=$HOME/auto-psych                      # your checkout: ALWAYS export this (_env.sh defaults to $HOME/auto-psych)
export OUTER_LIVE_SLURM_DIR=$REPO/scripts/outer_loop_live
sbatch $OUTER_LIVE_SLURM_DIR/setup.sbatch          # 40 min, 4 CPUs; builds $WORK_ROOT/venv
```

The log (`outer_live_setup_<jobid>.out`, in the directory you submitted from)
must end with `[setup] live import chain OK` and `[setup] done.`. Without
`OUTER_LIVE_SLURM_DIR` the job cannot find `_env.sh`.

`_env.sh` runs `set -euo pipefail`: `source` it only in a sub-shell (type
`bash` first), or a later failing command closes your login shell.

## 3. The config

Copy a preset, never edit it in place: `cp $REPO/scripts/outer_loop_live/pilot.yaml $REPO/my_pilot.yaml`.

| preset | launcher | settings |
|---|---|---|
| `pilot.yaml` | `run_pilot.sh` (one run) | 2 experiments × 10 people; 2 rounds × 3 proposals; 2,000 draws |
| `full_run.yaml` | `start_full_run.sh` (K parallel runs, default 3) | 3 × 40; 2 × 3; 3,000 draws |
| `hero_run.yaml` | `CONFIG=… start_full_run.sh` | 3 × 40; 4 × 7; 3,000 draws |

None of these matches the simulated checks (5 rounds × 6, 30-minute agents;
see [simulations_and_validation.md](simulations_and_validation.md)). Choose
deliberately.

Keys (read by `_pilot_config.py`):

| key | meaning |
|---|---|
| `run_label` | names the page URL (`/e<N>-<label>/`), the output directory `$WORK_ROOT/<label>/` and the sessions. **New label for every new run.** Ignored by `start_full_run.sh`, which uses `run1`…`runK`. |
| `experiments` | how many in sequence; each is its own deploy and its own study |
| `coding_agent`, `claude_auth` | `opencode` (default) or `claude`; with `claude`, `subscription` or `api` (required) |
| `prolific_mode` | `test` (the pilot preset): deploy, create a draft study, not published, then stop. `live`: publish, recruit, pay, model. `none`: deploy the page, no study, then stop. Missing key ⇒ `test`. |
| `confirm_live_recruitment` | must be `true` for `live`. `full_run.yaml` and `hero_run.yaml` say `live` without it, so they are refused until you choose. |
| `walltime`, `qos` | Slurm limit; `qos: long` above 2 days |
| `prolific.participants` | per experiment: the places recruited, the collection target **and** the design's N |
| `prolific.reward_per_hour` (cents) or `reward` (cents flat), `estimated_completion_time` (min) | pay |
| `prolific.name`, `description`, `completion_code`, `min_approval_rate` | study settings; use a distinct completion code per series |
| `prolific.completion_code_action` | `AUTOMATICALLY_APPROVE` (default) or `MANUALLY_REVIEW` |
| `modeling.inner_loop_iterations`, `inner_loop_candidates`, `draws`, `tune`, `chains` | rounds, proposals per round, MCMC (keep `chains` ≤ 4: the job has 4 CPUs, 64 GB) |
| `modeling.novelty_rmse_threshold`, `prune_dse_multiplier`, `candidate_parallelism`, `hints_file` | optional; defaults 0.002, 2.0, all at once, the built-in twelve angles |

**Cost** (printed by the launcher): pay per person = cents/hour × minutes / 60,
plus an estimated 33% Prolific fee (hard-coded; check your account). At
$12/h and 7 minutes: $1.40 a person; a pilot (2 × 10) ≈ $37; a full or hero
run (3 × 40) ≈ $223 per run, ≈ $670 for K = 3. Language-model costs come on
top and are not estimated; each experiment records its spend in
`token_usage_summary.json`.

**Time.** Per experiment: a few minutes of design, up to three 15-minute
attempts at the page, the deploy, **up to 3 hours of recruiting**, then the
model stage (length unknown; `full_run.yaml`'s comment says 12–15 hours for 3
experiments; a proposal's fit may now take up to 30 minutes, and agents that
hit a usage limit wait for it to reset). Set `walltime` generously.

## 4. Safety gates

| gate | what it stops |
|---|---|
| `confirm_live_recruitment: true` (launcher) and `--confirm-live-recruitment` (`run.py`) | one edited word spending money |
| cost summary, config check, Prolific token check, typed `yes` | typos and accidental launches (`CONFIRM=yes` skips the prompt: not for live runs) |
| preflight in `run_pilot.sh`: venv, `FIREBASE_TOKEN`, `PROLIFIC_API_TOKEN`, consent text; `_env.sh` stops without `bwrap` | failing hours into a job (the results token is checked later, before anything is deployed) |
| `--n-participants` is the only count; a rendered `prolific_config.yaml` with a different `total_available_places`, or none at all, stops `run.py` | recruiting a different number than the design assumed |
| results token (and, for `live`, Prolific's eligibility settings) checked first; the page is deployed, its session registered and checked live **before** the draft study is created, recorded and (live only) published | recruiting onto a broken or unprotected page; a failed deploy leaving a study behind |
| the deploy records the commit the code came from (from git, or from the record the launcher writes into the run copy) and refuses without one | a study that cannot be traced to its code |
| relaunch guard: an experiment whose `deployment/deployment_manifest.json` records a live study refuses `2_design`, `3_implement` and the deploy (`LiveStudyAlreadyRecorded`) | a second paid study for the same experiment |
| 3-hour give-up pauses an `ACTIVE` study | recruiting people whose data no experiment uses |
| raw download outside the repository (`run.py` refuses otherwise) | agents reading Prolific IDs |
| collection stops on an empty download or all-identical answers | modelling broken data |

## 5. No-cost rehearsal (all of it, before the first paid study)

In a sub-shell:

```bash
bash
export REPO=$HOME/auto-psych OUTER_LIVE_SLURM_DIR=$HOME/auto-psych/scripts/outer_loop_live
source $OUTER_LIVE_SLURM_DIR/_env.sh      # WORK_ROOT, VENV_PY, secrets
cd $REPO
"$VENV_PY" scripts/outer_loop_live/_pilot_config.py $REPO/my_pilot.yaml --render-only   # writes src/pipelines/outer_loop/projects/subjective_randomness/prolific_config.yaml
```

`--render-only` validates the config too, so a `live` config needs
`confirm_live_recruitment: true` here.

**R1. Offline dry run** (no network): staging, consent page, and the exact
study request, unsent.

```bash
AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/rehearsal-dry/data \
srun -p dev -t 10:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only --n-participants 10 \
  --deploy-target dry-run --prolific-mode test --run-label rehearsal-dry
"$VENV_PY" -m json.tool $WORK_ROOT/rehearsal-dry/data/subjective_randomness/experiment1/deployment/deployment_manifest.json
```

Check `metadata.prolific_payload`: `reward`, `total_available_places`,
`estimated_completion_time`, `filters`, `completion_codes`,
`external_study_url`. Delete `$WORK_ROOT/rehearsal-dry` before rerunning.

**R2. Simulated end-to-end run** (no Firebase, no Prolific; language-model
tokens only). The same job script with simulated participants, from a run
copy made as `run_pilot.sh` makes it:

```bash
LABEL=rehearsal-sim
sed -e 's/--mode live/--mode simulated_participants/' \
    -e '/--deploy-target firebase/d' -e '/--prolific-mode/d' -e '/--firebase-project/d' \
    $OUTER_LIVE_SLURM_DIR/run_live.sbatch > $WORK_ROOT/run_sim.sbatch
WT=$WORK_ROOT/runs/$LABEL/repo; mkdir -p $WT
rsync -a --delete \
  --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees' \
  --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  $REPO/ $WT/ && touch $WT/.here
sbatch --job-name=sim_$LABEL --time=1-00:00:00 \
  --output=$WORK_ROOT/slurm_logs/%x_%j.out --error=$WORK_ROOT/slurm_logs/%x_%j.out \
  --export=ALL,RUN_LABEL=$LABEL,RUN_WORKTREE=$WT,AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/$LABEL/data,EXPERIMENTS=2,N_PARTICIPANTS=10,INNER_LOOP_ITERATIONS=2,INNER_LOOP_CANDIDATES=3,DRAWS=2000,TUNE=2000,CHAINS=4 \
  $WORK_ROOT/run_sim.sbatch
```

(The generated script passes `bash -n`.) It exercises design, the
agent-built page (built, not deployed), collection, the whole model stage,
the carry-over and experiment 2's design. Success: the log contains
`All experiments complete.`, and both experiments have `model_loop/report.md`
and `cognitive_models/models_manifest.yaml`. Open
`experiment1/experiment/index.html` in a browser.

**R3. Real Firebase deploy, no study.** Needs `FIREBASE_TOKEN` and
`AUTO_PSYCH_RESULTS_TOKEN`; click through the real page, consent included.

```bash
AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/rehearsal-fb/data \
srun -p dev -t 15:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only \
  --deploy-target firebase --prolific-mode none \
  --firebase-project auto-psych-2c5da --run-label rehearsal-fb
curl -s -o /dev/null -w '%{http_code}\n' https://auto-psych-2c5da.web.app/e1-rehearsal-fb/
```

This replaces the project's **default** Hosting site: never do it while a
pilot (which uses that site) is recruiting. Parallel full runs have their own
sites.

**R4. Prolific test mode.** In your config: `prolific_mode: test` and a fresh
`run_label`; run `CONFIG=$REPO/my_pilot.yaml bash scripts/outer_loop_live/run_pilot.sh`
and type `yes`. It designs, builds, deploys and then creates an **unpublished
draft**, then stops (experiment 1 only). This is the first rehearsal through
the launcher's run copy: check that `deployment_manifest.json` has a
`git_commit` and `metadata.code_provenance` naming your checkout. Preview it from the Prolific
dashboard, or open the URL with `?PROLIFIC_PID=test123`. Delete the draft
afterwards.

## 6. Launch

Launchers run on a login node; they only check, copy and submit.

**One run (pilot):** in your config set `prolific_mode: live`,
`confirm_live_recruitment: true`, a new `run_label`, participants, pay and
`walltime`. Then:

```bash
export REPO=$HOME/auto-psych
CONFIG=$REPO/my_pilot.yaml bash $REPO/scripts/outer_loop_live/run_pilot.sh
```

Read the summary (the `LIVE` banner, reward, eligibility, participants,
`PROLIFIC TOTAL`, `prolific token : OK`, URLs, output directory) and type
`yes` only if all of it is right. Keep the printed job id, log path
(`$WORK_ROOT/slurm_logs/pilot_<label>_<jobid>.out`) and stop instructions.

**K parallel runs:**

```bash
CONFIG=$REPO/my_full_run.yaml K=3 bash $REPO/scripts/outer_loop_live/start_full_run.sh
```

It prints the cost per run and lists the earlier `$WORK_ROOT/run<i>` and
`$WORK_ROOT/runs/run<i>` directories it will **delete**; nothing is deleted
before `yes`. Collect earlier results first (§ 10). Deleting a run's
directories also deletes the record the relaunch guard reads, so never reuse
an index whose study is still open. `RUNS="2 3"` launches only those indices.
Each run gets its own copy of the code, output tree, Hosting site
(`https://auto-psych-2c5da-run<i>.web.app/e<N>-run<i>/`), studies, and log
(`$WORK_ROOT/slurm_logs/outer_live_run<i>_<jobid>.out`).

Each launch copies your **current working tree** (no commit needed) and
records, in the copy's `code_provenance.json`, the commit it came from and
whether the tree had uncommitted or untracked changes. The deployment
manifest copies them (`git_commit`, `git_dirty`, `metadata.code_provenance`).

## 7. Monitoring

No more than about once a minute, and never in a loop:

- `squeue --me`; `tail -n 40 <log>` for stage banners and `[error]` lines.
- `…/experiment<N>/data/observability.log`: one line per Prolific check, e.g.
  `Prolific poll: study_id='…' completed=7 target=10`.
- The **Prolific dashboard**: places filled, returns, median time.
- The **live dashboard** (`src/monitor/`) reads answers from Firestore as they
  arrive. It flags a participant with 3+ trials all on one side, and a
  session (10+ trials) where ≤ 5% or ≥ 95% of choices went left. The pipeline
  itself only aborts when *every* answer is identical, after the study ends,
  so this is how you catch a broken button in time. Run it on your own
  computer (Sherlock has no `gcloud`):

  ```bash
  uv sync
  gcloud auth application-default login
  gcloud auth application-default set-quota-project auto-psych-2c5da
  # PROLIFIC_API_TOKEN in this checkout's .secrets, for recruitment counts
  rsync -a --prune-empty-dirs --include='*/' --include='deployment_manifest.json' --exclude='*' \
    <sunet>@login.sherlock.stanford.edu:/scratch/users/<sunet>/auto-psych/outer_loop_live/ ./live_manifests/
  uv run python -m src.monitor.server --data-root ./live_manifests    # http://127.0.0.1:8001
  ```

  Re-run the `rsync` after each deploy. Keep the default host: the dashboard
  has no login and shows Prolific IDs. (Not run against live services.)

## 8. Stopping

1. **Pause or stop every published study in the Prolific dashboard.** Only
   this stops recruiting and paying; `scancel` does not, and a crashed or
   cancelled job pauses nothing.
2. `scancel <jobid>`.
3. Delete leftover `test` drafts.

The page stays online; people already in the study can still submit.

## 9. Recovering without paying again

Never relaunch a run whose experiment has a live study: the job fails with
`LiveStudyAlreadyRecorded`. Rerun only the stages you need with
`RESUME_AGENTS` (stages separated by `:`); leaving out `3_implement` never
redeploys or recruits.

```bash
export REPO=$HOME/auto-psych OUTER_LIVE_SLURM_DIR=$REPO/scripts/outer_loop_live
WORK_ROOT=$SCRATCH/auto-psych/outer_loop_live; LABEL=<label>
sbatch --job-name=resume_$LABEL --time=12:00:00 \
  --output=$WORK_ROOT/slurm_logs/%x_%j.out --error=$WORK_ROOT/slurm_logs/%x_%j.out \
  --export=ALL,RUN_LABEL=$LABEL,RUN_WORKTREE=$WORK_ROOT/runs/$LABEL/repo,AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/$LABEL/data,EXPERIMENT=1,N_PARTICIPANTS=10,PROLIFIC_MODE=none,RESUME_AGENTS=4_collect:5_model_loop,DRAWS=2000,TUNE=2000,CHAINS=4,INNER_LOOP_ITERATIONS=2,INNER_LOOP_CANDIDATES=3 \
  $OUTER_LIVE_SLURM_DIR/run_live.sbatch
```

- `PROLIFIC_MODE=none` is safe: collection in `--mode live` reads the study
  id from `experiment/config.json`. Keep `N_PARTICIPANTS` equal to the
  config's `participants`.
- Rerunning `4_collect` waits for the target again (up to 3 hours) and
  downloads everything again.
- `5_model_loop` always restarts cleanly from the model set it first
  started with.
- The run copy is a snapshot: to use fixed code, rsync it in first (as in R2).
  The rsync replaces the study settings `run_pilot.sh` rendered into the copy
  and deletes its commit record. Render the settings again with
  `"$VENV_PY" $WORK_ROOT/runs/$LABEL/repo/scripts/outer_loop_live/_pilot_config.py <your.yaml> --render-only`
  before any run with a Prolific mode other than `none`, and, before any
  deploy, record the commit again from your checkout:
  `cd $REPO && "$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout $REPO --copy $WORK_ROOT/runs/$LABEL/repo`.
- To run the remaining experiments, submit the same command without
  `RESUME_AGENTS`, with `EXPERIMENTS=<next>-<last>`, `PROLIFIC_MODE=live`
  and `CONFIRM_LIVE_RECRUITMENT=1`; for a parallel run `run<i>` also
  `AUTO_PSYCH_HOSTING_SITE=auto-psych-2c5da-run<i>` (without it the deploy
  goes to the default site). This publishes new studies and pays.
- A second study for the same experiment: stop the first in Prolific, then
  set `PUBLISH_ANOTHER_PROLIFIC_STUDY=1`; the old manifest is kept as
  `deployment_manifest.superseded-<time>.json`.

## 10. Where the data are, and collecting them

```
$WORK_ROOT/<label>/data/subjective_randomness/
├── raw_collected/experiment<N>_responses.csv   everything /results returned: CONTAINS PROLIFIC IDs
└── experiment<N>/                              design/, experiment/, deployment/, data/, model_loop/, cognitive_models/
$WORK_ROOT/runs/<label>/repo/                   the run's copy of the code
$WORK_ROOT/slurm_logs/                          job logs
```

`$SCRATCH` is purged after 90 days without modification: move what you keep
(raw data only to access-controlled storage such as `$OAK`, as your IRB
protocol allows).

`scripts/outer_loop_live/collect_results.sh` copies each run's
`data/subjective_randomness/` tree into the repository and **removes every
Prolific ID**: it drops ID columns from every CSV, redacts every 24-character
hex token in text files, and fails if any survives. It skips `.nc` fits and
the run copies, and writes `SUMMARY.md`. It copies `raw_collected/` too, with
the ID column dropped.

```bash
cd $REPO
PY=$SCRATCH/auto-psych/outer_loop_live/venv/bin/python RUNS="run1 run2 run3" \
DEST=data/results/human_experiment_<month> \
  bash scripts/outer_loop_live/collect_results.sh        # runs in a short dev job
# a pilot: RUNS="<label>" (names starting with pilot are skipped unless named)
```

Set `DEST` to a **new** directory: the default `data/results/human_experiment/`
holds the paper's human study, and `--overwrite` copies over it (it does not
clear files already there). The default
`PY` (`$REPO/.venv`) may not exist on the cluster.

## 11. Prolific IDs (24 hex characters) identify people

- Never commit, share, paste or send to a chatbot or coding agent any file
  that may hold one: `raw_collected/`, `experiment/config.json`,
  `deployment_manifest.json`, logs, the dashboard. Share only the output of
  `collect_results.sh`.
- Agents never see them: `data/responses.csv` has only the five raw
  columns, and `run.py` refuses an output tree inside the repository. If you
  call `run.py` by hand, set `AUTO_PSYCH_OUTPUT_DIR` outside it.
- `python -m src.viewer.freeze` copies transcripts verbatim: freeze scrubbed
  runs only.
- Sherlock is approved for Low and Moderate Risk data. If you are unsure how
  your IRB classifies these data, ask before collecting.

## Docs and code that disagree

| where | says | code does |
|---|---|---|
| `_env.sh` | Python 3.12 | the repository pins 3.11; `setup.sbatch` installs pinned `pymc==5.28.5`, `arviz<1` instead (not tested here) |

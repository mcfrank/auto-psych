# Running a live experiment (Prolific + Firebase) on Sherlock

This runbook takes you from a fresh checkout to a finished live run:
credentials, setup, config, the safety gates, a no-cost rehearsal, the real
launch, monitoring, stopping, recovering, and collecting the data. Every
command was checked against the scripts and command-line interfaces on
27 September 2026. **Nothing on this page was actually run against Prolific or
Firebase while writing it.** Commands that talk to those services are
described from the code.

Placeholders: `$REPO` is your checkout, `$WORK_ROOT` is where the live runs
write (default `$SCRATCH/auto-psych/outer_loop_live`), and `<label>` is a run
label you choose.

---

## 0. Read this first: known problems as of 27 September 2026

### Problems that stop a run

**A. The model stage crashes on collected data.** Confirmed by reproducing the
failing step in isolation.

- The `/results` endpoint (`functions/index.js`) returns eight columns:
  `participant_id, participant_id_str, trial_index, sequence_a, sequence_b,
  chose_left, chose_right, model`. The simulated collectors add `chose_right`
  and `model` too.
- `4_collect` writes these to `data/responses.csv` unchanged.
  `_pooled_response_rows` then copies them into `model_loop/responses.csv`
  (`src/pipelines/outer_loop/model_loop_runner.py`).
- The candidate brief refuses any column beyond the five raw ones
  (`_write_candidate_context` in `src/pipelines/inner_loop/candidate_agent.py`),
  with the error:

  ```
  ValueError: .../model_loop/responses.csv has column(s) ['participant_id_str', 'chose_right', 'model'] beyond the raw ones [...]
  ```

- So `5_model_loop` stops at the first round of new models. In a live run that
  is **after the participants have been paid**. The data themselves are safe
  on disk and can be re-modelled once this is fixed (see § 10).
- **Privacy side effect.** `participant_id_str` is the participant's Prolific ID.
  As things stand, it is copied into the file the critique agent is pointed to,
  and that agent sends what it reads to an outside language-model provider.
  The fix must *drop* the extra columns (keep only the five raw ones), not
  merely tolerate them.
- The holdout simulations are not affected: they strip the extra columns
  themselves (`src/subjective_randomness/holdout_data.py`).

**B. The live launchers do not load `bubblewrap`.** Found by reading the code;
not run.

- Every coding agent (`3_implement`, critique, proposals) runs in a
  bubblewrap sandbox. It raises `Sandboxed agents need bubblewrap on PATH (on
  Sherlock: ml load system bubblewrap)` if `bwrap` is missing.
- `scripts/outer_loop_live/_env.sh` runs `ml purge` and never loads it.
  `scripts/subjective_randomness/slurm/_env.sh` does, at line 30.
- `/usr/bin/bwrap` does not exist on Sherlock. Unless your own shell startup
  puts `bwrap` on the `PATH` in a way that survives `ml purge`, a live job will
  fail at `3_implement`. That is **before** anything is deployed or paid for.
- The fix is one line in `scripts/outer_loop_live/_env.sh`:
  `ml load system bubblewrap`.

### Surprises that cost money or data if you don't know them

| # | What | Consequence | What to do |
|---|---|---|---|
| C | **Reusing a run label, or relaunching after a crash, re-runs every stage**, including creating and publishing a *new* Prolific study. The job always passes `--resume`, and nothing records "this experiment was already deployed". This holds for multi-experiment runs too: relaunching a 2-experiment run that died in experiment 2 redoes experiment 1. | Paying twice. | Use a new `run_label` for every launch. Recover a stalled run only with `RESUME_AGENTS` (§ 10). |
| D | **After 2 hours the pipeline stops waiting**, even if fewer participants have finished. It models whatever data exist, **but the Prolific study stays open**. Nothing in the code pauses or stops a study. | Late participants are recruited and paid but not used by that experiment. | Watch the study. Stop it in the Prolific dashboard when the pipeline moves on (§ 8). |
| E | `start_full_run.sh` **deletes the output and run-copy directories of the runs it is about to launch** (`$WORK_ROOT/run<i>`, `$WORK_ROOT/runs/run<i>`) *before* it asks you to type `yes`. | Answering "no" still deletes earlier `run1`…`runK` results. | Collect earlier results first (§ 11), or use `RUNS=` to pick other indices. |
| F | When calling `run.py` directly (not via the launchers), **the number of Prolific places comes from the project's `prolific_config.yaml`, not from `--n-participants`**. The committed file says 40 places, 5 minutes, $12/h. `--n-participants` only sets the design's N and the waiting target. | E.g. `--n-participants 5` recruits and pays 40 people, and the pipeline moves on after 5. | Use the launchers. They write that file from your config, so the two numbers agree. |
| G | The cost summary covers **Prolific only**. Language-model spending is recorded afterwards (`token_usage_summary.json`) but not estimated beforehand. | — | Check your opencode/Gemini or Anthropic billing separately. |
| H | Payment is **automatic on completion** (`AUTOMATICALLY_APPROVE`). Set `completion_code_action: MANUALLY_REVIEW` under `prolific:` to review first. | Low-effort submissions are paid. | Deliberate choice: data quality is handled in analysis, not by withholding pay. |

### Documentation that disagrees with the code

| Where | Says | Code does |
|---|---|---|
| `scripts/outer_loop_live/README.md`, `run_pilot.sh` comments | `sbatch scripts/outer_loop_live/setup.sbatch` from the repo root | The job looks for `_env.sh` in `$OUTER_LIVE_SLURM_DIR`, else the directory you submitted from, so that command fails. Use the command in § 2. The same applies to submitting `run_live.sbatch` by hand. |
| `scripts/outer_loop_live/_env.sh` | `REPO` defaults to `$HOME/auto-psych` | If your checkout is elsewhere, the launchers copy and read secrets from the wrong place. Always `export REPO=...`. |
| `.secrets.example` | Lists three keys | Also needs `AUTO_PSYCH_RESULTS_TOKEN`. The deploy stops without it, before anything is staged or any study is created. |
| `scripts/outer_loop_live/README.md` | `.firebaserc` "already present" | Only `.firebaserc.example` is in the repository. Harmless: the launchers always pass `--firebase-project`. |
| `pilot.yaml` comment | Default is `prolific_mode: test` | The file ships with `prolific_mode: live` and no `confirm_live_recruitment`. The launcher refuses it until you choose (safe, but confusing). |
| `start_full_run.sh` header | "~$480" for the full run | With the current `full_run.yaml` (7 min, $12/h, 40 × 3 × 3) the estimate is **≈ $670** (§ 3). |
| `scripts/outer_loop_live/README.md`, isolation table | Parallel runs share one site, one path each | Each parallel run now deploys to **its own Firebase Hosting site**, `https://<firebase_project>-run<i>.web.app/e<N>-run<i>/`. A pilot uses the project's default site. |
| `hero_run.yaml`, main `README.md` | "7 candidates, one exploration lens each"; starting models are "the best models from three human replicates" | With 7 per round: 4 explore, 2 improve-the-best, 1 improve-another. The starting models are four literature models. |
| `run.py --help` | Novelty threshold default "0.02" | The actual default is **0.002** (`DEFAULT_NOVELTY_RMSE_THRESHOLD` in `model_zoo.py`). |
| `scripts/outer_loop_live/README.md` | `GOOGLE_API_KEY` is not needed for a live human run | With the default coding agent (opencode + Gemini), the agents' Gemini login is taken from `GOOGLE_API_KEY` (copied to `GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY` by `_env.sh`), unless opencode has other credentials. I could not check how opencode is logged in on your account. Assume you need it. |
| `_env.sh` | Python 3.12 environment | The repository pins Python 3.11 (`.python-version`) and warns against newer versions. The live environment gets around this by installing a fixed list (`pymc==5.28.5`, `arviz<1`, …) rather than the locked environment. I did not test that environment. |

---

## 1. Accounts and credentials

You need:

- A **Sherlock** account. The runs are Slurm jobs, CPU only, on partition `normal`.
- A **Prolific researcher account with funds**, and an API token.
- Access to the **Firebase project `auto-psych-2c5da`**, and a non-interactive
  deploy token made with `firebase login:ci` on a machine with a browser.
- A login for the **coding agent**: a Gemini API key for opencode (the
  default), or a logged-in Claude Code under `~/.claude` if you set
  `coding_agent: claude`. `_env.sh` loads the `opencode` and `claude-code`
  modules on Sherlock.
- **IRB approval** for the exact consent text in `templates/consent.txt`. It is
  shown verbatim as the first page. Check it before every study; it names the
  lab and says participants' anonymity is assured.
- For the live dashboard (optional): on **your own computer**, the Google Cloud
  SDK logged in with read access to that Firestore project. There is no
  `gcloud` module on Sherlock.

Credentials go in **`$REPO/.secrets`**, one `KEY=value` per line. It is
gitignored and must never be committed, copied or printed. `.secrets.example`
is the template. `_env.sh` reads the file and exports every key into the job.
Run `chmod 600 $REPO/.secrets`.

| key | used for | needed when |
|---|---|---|
| `PROLIFIC_API_TOKEN` | creating, publishing and polling studies; the launcher's token check | Prolific mode `test` or `live` |
| `FIREBASE_TOKEN` | `firebase deploy` from a compute node | any `firebase` deploy |
| `AUTO_PSYCH_RESULTS_TOKEN` | shared secret protecting `/results` and `/register_session`. Make one with `openssl rand -hex 32` and use **the same value** for every deploy and collection. The deploy writes it into `functions/.env`. | any `firebase` deploy, and every live collection |
| `GOOGLE_API_KEY` | opencode's Gemini login (see § 0); also the simulated "language model as participant" mode | default coding agent |

The sandboxed agents never see the Prolific, Firebase or results tokens. They
get only their own API login (`AGENT_ENV_NAMES` / `BACKEND_ENV` in
`src/runtime/agent_sandbox.py`).

## 2. One-time environment setup on Sherlock

```bash
# 1. The checkout. Code belongs in $HOME or $GROUP_HOME; job output goes to
#    $SCRATCH (the launchers default to $SCRATCH/auto-psych/outer_loop_live).
export REPO=$HOME/auto-psych          # wherever your checkout is: ALWAYS export this
cd $REPO && git status                # on the branch you mean to run

# 2. Build the shared environment (Python venv, Firebase functions packages,
#    firebase CLI) as a Slurm job. OUTER_LIVE_SLURM_DIR tells the job where
#    _env.sh is.
export OUTER_LIVE_SLURM_DIR=$REPO/scripts/outer_loop_live
sbatch $OUTER_LIVE_SLURM_DIR/setup.sbatch
squeue --me                           # check again after a minute or more, not in a loop
# The log (setup's %x_%j.out, in the directory you submitted from) must end with:
#   [setup] live import chain OK
#   [setup] done.
```

The environment lands in `$WORK_ROOT/venv` (40-minute job, 4 CPUs, 16 GB).
Rebuild it only when dependencies change. If you use a non-default
`WORK_ROOT`, export it before every command on this page.

**Shell caution.** `_env.sh` turns on `set -euo pipefail`. If you `source` it
in your login shell, any later failing command closes that shell. Source it
only inside a sub-shell: type `bash` first, and `exit` when done.

## 3. Choosing and editing the config

There are three config files in `scripts/outer_loop_live/`, all in the same format:

| file | launcher | preset |
|---|---|---|
| `pilot.yaml` | `run_pilot.sh` (one run) | 2 experiments × 10 participants, 2 rounds × 3 proposals |
| `full_run.yaml` | `start_full_run.sh` (K parallel runs, default 3) | 3 experiments × 40 participants, 2 rounds × 3 proposals |
| `hero_run.yaml` | `CONFIG=…/hero_run.yaml start_full_run.sh` | 3 experiments × 40 participants, 4 rounds × 7 proposals |

Copy one instead of editing it in place, for example
`cp scripts/outer_loop_live/pilot.yaml $REPO/my_pilot.yaml`, and pass
`CONFIG=$REPO/my_pilot.yaml`. Keys, as read by
`scripts/outer_loop_live/_pilot_config.py`:

| key | meaning |
|---|---|
| `project` | `subjective_randomness` |
| `run_label` | names the run: URL path `/e<N>-<label>/`, output dir `$WORK_ROOT/<label>/`, session ids. **New label for every launch** (§ 0 C). Ignored by `start_full_run.sh`, which uses `run1`…`runK`. |
| `experiments` | how many experiments in sequence. Each is its own deploy and its own Prolific study. |
| `coding_agent` | `opencode` (default) or `claude` |
| `prolific_mode` | `test`: create a **draft** study (not published), deploy, and stop. `live`: publish, recruit, pay, run everything. `none`: deploy only, no study. |
| `confirm_live_recruitment` | must be `true` for `live` (§ 4) |
| `firebase_project` | `auto-psych-2c5da` |
| `walltime`, `qos` | Slurm time limit (`D-HH:MM:SS`); `qos: long` for more than 2 days (up to 7) |
| `prolific.participants` | participants per experiment. Becomes the Prolific places, the waiting target **and the N the design assumes**. |
| `prolific.reward_per_hour` (cents) *or* `prolific.reward` (cents flat) | pay. Hourly pay is converted using `estimated_completion_time`. |
| `prolific.estimated_completion_time` | minutes. 64 trials take about 6–7 minutes according to the configs' own comments. |
| `prolific.name`, `description`, `device_compatibility`, `completion_code` | what participants see on Prolific. Use a distinct completion code per study series. |
| `prolific.min_approval_rate` | approval-rate floor in percent (default 98). US residence and English fluency are fixed in code, and their Prolific IDs are re-checked against Prolific before a live study is created. |
| `prolific.completion_code_action` (optional) | `AUTOMATICALLY_APPROVE` (default) or `MANUALLY_REVIEW` |
| `modeling.inner_loop_iterations`, `inner_loop_candidates` | rounds per experiment, proposals per round |
| `modeling.draws`, `tune`, `chains` | MCMC settings. Keep `chains` ≤ 4: the job gets 4 CPUs. |
| `modeling.hints_file`, `novelty_rmse_threshold`, `prune_dse_multiplier`, `candidate_parallelism` | optional overrides for the exploration angles, the near-duplicate threshold (0.002), the pruning multiplier (2.0), and how many proposing agents run at once |

The job itself (`run_live.sbatch`) requests partition `normal`, 4 CPUs and 64 GB.

**Cost.** The launcher prints this formula: reward per person = round(cents/hour
× minutes / 60), plus an estimated Prolific fee of 33%. **Confirm the current fee
in your Prolific account**; the 33% is hard-coded in `_pilot_config.py`.

| config | per person | per experiment (reward + fee) | total |
|---|---|---|---|
| `pilot.yaml` (10 people, 7 min, $12/h, 2 experiments) | $1.40 | $14.00 + $4.62 = $18.62 | **$37.24** |
| `full_run.yaml` / `hero_run.yaml` (40 people, 3 experiments) | $1.40 | $56.00 + $18.48 = $74.48 | $223.44 per run, **≈ $670 for K = 3** |

Language-model costs come on top (§ 0 G).

**Time.** Each experiment takes a few minutes of design, up to three
15-minute attempts at building the page, the deploy, **up to 2 hours of
recruiting**, then the model stage. The model stage's length depends on the
number of rounds and proposals, and I could not estimate it from the code;
`full_run.yaml`'s comment says a 3-experiment run needs about 12–15 hours. Set
`walltime` generously.

## 4. The safety gates, and why each exists

| gate | where | what it stops |
|---|---|---|
| `prolific_mode` defaults to `test` when the key is missing | `_pilot_config.py` | a forgotten key cannot recruit anyone |
| `confirm_live_recruitment: true` required for `live` | `_pilot_config.py` (launcher) | a single edited word (`test` → `live`) cannot spend money |
| `--confirm-live-recruitment` required for `--prolific-mode live` | `run.py`, checked before anything else | the same, for anyone calling `run.py` or `run_live.sbatch` directly. The launcher passes it only when the config says `true`. |
| Preflight: venv built, `FIREBASE_TOKEN`, `PROLIFIC_API_TOKEN`, consent text present | `run_pilot.sh` | failing hours into a job. (It does *not* check `AUTO_PSYCH_RESULTS_TOKEN` or `bwrap`.) |
| Config validation + **cost summary** + Prolific token check (a read-only `GET /users/me/`) | `_pilot_config.py --check` | typos, zero pay, a bad approval floor, an invalid token |
| **Typed `yes`** (the prompt text says LIVE / TEST / deploy-only) | `run_pilot.sh`, `start_full_run.sh` | launching by accident. `CONFIRM=yes` skips it; don't use that for live runs. |
| Results token required before staging | `deployment/firebase.py` | publishing an endpoint anyone could read participant data from |
| Session registration, then **page-is-live check**, before any study is published | `deployment/local.py`, `firebase.py` | recruiting people onto a broken (404) page, or onto a page whose data would be rejected |
| Prolific eligibility IDs re-checked against Prolific before a live study is created | `deployment/prolific.py` | recruiting the wrong population if Prolific renumbers its filters |
| Refuses zero pay | `compute_reward_cents` | a study that pays nothing |
| `test` mode runs only the first experiment and stops after deploy | `run.py` | a "test" that goes on to collect or model |
| Collection refuses to continue on an empty download, a failed download or all-identical responses | `orchestrator.py`, `collect.py` | modelling empty or broken data as if it were real |

## 5. No-cost rehearsal (do all of it before the first paid study)

The code supports four rehearsals, from cheapest to most complete. Run all
of them in a sub-shell (`bash`) with the environment loaded:

```bash
bash
export REPO=$HOME/auto-psych OUTER_LIVE_SLURM_DIR=$HOME/auto-psych/scripts/outer_loop_live
source $OUTER_LIVE_SLURM_DIR/_env.sh      # sets WORK_ROOT, VENV_PY, loads secrets
cd $REPO
```

### R1. Offline dry run: no network, no cost

This checks the staging, the consent gate and the manifest, and builds the
exact Prolific study request without sending it. It uses a tiny built-in
experiment rather than an agent-built one.

```bash
AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/rehearsal-dry/data \
srun -p dev -t 10:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only \
  --deploy-target dry-run --prolific-mode test --run-label rehearsal-dry
```

Then inspect the study that *would* be created:

```bash
"$VENV_PY" -m json.tool \
  $WORK_ROOT/rehearsal-dry/data/subjective_randomness/experiment1/deployment/deployment_manifest.json
```

Check `metadata.prolific_payload`: `reward` (cents), `total_available_places`,
`estimated_completion_time`, `filters`, `completion_codes`,
`external_study_url`. The staged page is under `…/experiment1/deployment/public/`.

- The payload uses `src/pipelines/outer_loop/projects/subjective_randomness/prolific_config.yaml`
  from the checkout. As committed, that file is a stale copy (40 places,
  5 min). To see *your* config's numbers, write it into that file first with
  `"$VENV_PY" scripts/outer_loop_live/_pilot_config.py <your.yaml> --render-only`.
  That overwrites the tracked file; `git checkout` it afterwards if you like.
  `--render-only` still validates the config, so a `live` config needs
  `confirm_live_recruitment: true` or it stops.
- Re-running with the same output directory fails with "experiment directory
  already exists". Delete `$WORK_ROOT/rehearsal-dry` first.

### R2. Simulated end-to-end run: no Firebase, no Prolific, costs language-model tokens only

**This is the rehearsal that would have caught problem A.** It runs the full
loop with simulated participants, through the same job script as a live run.
There is no ready-made launcher, so make a simulated copy of the live job
script and a run copy of the repository, the same way `run_pilot.sh` does:

```bash
LABEL=rehearsal-sim
sed -e 's/--mode live/--mode simulated_participants/' \
    -e '/--deploy-target firebase/d' -e '/--prolific-mode/d' -e '/--firebase-project/d' \
    $OUTER_LIVE_SLURM_DIR/run_live.sbatch > $OUTER_LIVE_SLURM_DIR/run_sim.sbatch
# (the result was syntax-checked with bash -n; delete run_sim.sbatch when done, it is not tracked)
WT=$WORK_ROOT/runs/$LABEL/repo; mkdir -p $WT
rsync -a --delete \
  --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees' \
  --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  $REPO/ $WT/ && touch $WT/.here      # same copy run_pilot.sh makes
sbatch --job-name=sim_$LABEL --time=1-00:00:00 \
  --output=$WORK_ROOT/slurm_logs/%x_%j.out --error=$WORK_ROOT/slurm_logs/%x_%j.out \
  --export=ALL,RUN_LABEL=$LABEL,RUN_WORKTREE=$WT,AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/$LABEL/data,EXPERIMENTS=2,N_PARTICIPANTS=10,INNER_LOOP_ITERATIONS=2,INNER_LOOP_CANDIDATES=3,DRAWS=2000,TUNE=2000,CHAINS=4 \
  $OUTER_LIVE_SLURM_DIR/run_sim.sbatch
```

`--mode simulated_participants` draws responses from the current models'
priors: no browser, no Firebase, no Prolific. It exercises design, the
agent-built page (built, not deployed), collection, the whole model stage,
carrying models over, and the experiment-2 design. Success means the log contains
`All experiments complete.` and both `experiment1/` and `experiment2/`
contain `model_loop/report.md` and `cognitive_models/models_manifest.yaml`.
Open the agent-built `experiment1/experiment/index.html` in a browser too.

### R3. Real Firebase deploy, no Prolific study

This puts the real page online, so you can open it and click through it
yourself, including the consent page. It needs `FIREBASE_TOKEN` and
`AUTO_PSYCH_RESULTS_TOKEN`. It creates no Prolific study and costs nothing on
Prolific; I could not check the Firebase billing plan.

```bash
AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/rehearsal-fb/data \
srun -p dev -t 15:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only \
  --deploy-target firebase --prolific-mode none \
  --firebase-project auto-psych-2c5da --run-label rehearsal-fb
curl -s -o /dev/null -w '%{http_code}\n' https://auto-psych-2c5da.web.app/e1-rehearsal-fb/
```

This deploys to the project's **default** site. A Firebase Hosting deploy
replaces the whole site with the contents of the deploying checkout's
`public/` folder, so do not do this while a pilot on the default site is
recruiting: its page would disappear. (Parallel full runs use their own sites.)
The same caution applies to two pilots at once.

### R4. Prolific test mode: the real agent-built page plus a draft study

In your copy of `pilot.yaml`, set `prolific_mode: test` and a fresh
`run_label`, then run `CONFIG=$REPO/my_pilot.yaml bash scripts/outer_loop_live/run_pilot.sh`
and type `yes` at the `TEST:` prompt.

The job designs the stimuli, has the agent build the page, deploys it, creates
an **unpublished draft** study and stops. Nothing is collected or modelled, and
only experiment 1 runs. Preview the study from the Prolific dashboard, or open
the experiment URL with a made-up `?PROLIFIC_PID=test123`. Delete the draft in
Prolific afterwards. Use a different `run_label` for the live launch (§ 0 C).

## 6. The real launch

**Pilot (one run):**

1. In your config: `prolific_mode: live`, `confirm_live_recruitment: true`, a new
   `run_label`, the participants and pay you want, and a `walltime` that covers
   every experiment.
2. Launch (on a login node; the launcher only validates, copies and submits):
   ```bash
   export REPO=$HOME/auto-psych
   CONFIG=$REPO/my_pilot.yaml bash $REPO/scripts/outer_loop_live/run_pilot.sh
   ```
3. Read the printed summary: the `LIVE` banner, study name, reward per person
   and effective hourly rate, eligibility, participants, cost per experiment and
   **estimated total**, `prolific token : OK`, the experiment URLs and output
   directory. Type `yes` only if all of it is right.
4. It prints the job id, the log path
   (`$WORK_ROOT/slurm_logs/pilot_<label>_<jobid>.out`), the data path and the
   stop instructions. **Keep that output.**

**Full or hero run (K parallel runs):**

```bash
export REPO=$HOME/auto-psych
CONFIG=$REPO/scripts/outer_loop_live/hero_run.yaml K=3 bash $REPO/scripts/outer_loop_live/start_full_run.sh
```

- It validates the config and prints the cost **per run**. It then writes the
  study settings into the project's `prolific_config.yaml` in your checkout,
  **deletes `run1…runK`'s previous directories** (§ 0 E), and asks for `yes`.
- Each run `i` gets its own copy of the repository, its own output tree
  `$WORK_ROOT/run<i>/data/`, its own Firebase site and URL
  (`https://auto-psych-2c5da-run<i>.web.app/e<N>-run<i>/`), its own Prolific
  study per experiment, and the log `$WORK_ROOT/slurm_logs/outer_live_run<i>_<jobid>.out`.
- `RUNS="2 3"` (re)launches only those indices.
- The per-run site is created automatically if it does not exist; I could not
  check the Firebase project's site quota.

Each launch copies **your current working tree** into the run copy. There is
no commit or version check, so make sure the checkout is on the code you
intend to run. The deployment manifest records the commit and whether the tree
had uncommitted changes (`git_commit`, `git_dirty`).

## 7. Watching it run

At most every minute or so (Sherlock asks users not to poll the scheduler or
logs in a loop):

- `squeue --me`: is the job running?
- `tail -n 40 $WORK_ROOT/slurm_logs/<log>`: the stage banners
  (`Experiment 1 / Agent 2_design`, `… / Deployment (firebase)`, `[ok] …`,
  `[deploy] Registered collection session …`) and any `[error]`.
- `$WORK_ROOT/<label>/data/subjective_randomness/experiment<N>/data/observability.log`:
  during collection, one line per check, e.g.
  `Prolific poll: study_id='…' completed=7 target=10`, and the timeout
  message if it happens.
- **The Prolific dashboard**: one study per experiment, published by the job.
  Watch places filled, returns, time-outs and the median completion time, which
  should be near your `estimated_completion_time`.
- **The live dashboard** (`src/monitor/`) reads each participant's data from
  Firestore as it arrives, and Prolific's counts. Its main job is to catch
  **degenerate data** early:
  - a participant flagged **degenerate**: 3 or more valid trials, all on the
    same side;
  - a **session warning**: over 10 or more trials, 5% or fewer, or 95% or more,
    of all choices went left.

  The pipeline itself aborts only if *every* response is identical, and only
  after the study has finished. The dashboard is how you catch a
  button-mapping bug or a wave of click-through participants while you can still
  pause the study.

  Run it **on your own computer** (Sherlock has no `gcloud`):
  ```bash
  # once, on your computer, in a checkout of the same branch:
  uv sync
  gcloud auth application-default login
  gcloud auth application-default set-quota-project auto-psych-2c5da
  # put PROLIFIC_API_TOKEN in that checkout's .secrets (for recruitment counts)
  # fetch just the deployment manifests from Sherlock (they tell it what to watch):
  rsync -a --prune-empty-dirs --include='*/' --include='deployment_manifest.json' --exclude='*' \
    <sunet>@login.sherlock.stanford.edu:/scratch/users/<sunet>/auto-psych/outer_loop_live/ ./live_manifests/
  uv run python -m src.monitor.server --data-root ./live_manifests
  # open http://127.0.0.1:8001; the page refreshes itself every 15 s
  ```
  Re-run the `rsync` after each new experiment is deployed. Keep the default
  host `127.0.0.1`: the dashboard has no login and shows Prolific IDs.

  I did not run the monitor against live services. The commands follow its
  code and the help text in `src/monitor/sources.py`.

## 8. Stopping

**Order matters.**

1. **Stop or pause the study in the Prolific dashboard**, every study the run
   has published. This is the only thing that stops recruiting and paying.
   `scancel` does **not** touch a published study. Nothing in the code pauses
   or stops a study, including after the 2-hour timeout (§ 0 D).
2. `scancel <jobid>` stops the pipeline (waiting, modelling, agents).
3. Delete any leftover `test`-mode draft studies in Prolific.

The deployed page stays online after the job ends. Participants who already
started can still submit, and `/submit` keeps accepting data for a registered
session.

## 9. Where the data land

```
$WORK_ROOT/                                  (default $SCRATCH/auto-psych/outer_loop_live)
├── slurm_logs/                              job logs
├── venv/                                    shared Python environment
├── runs/<label>/repo/                       the run's copy of the code (safe to delete afterwards)
└── <label>/data/subjective_randomness/      (<label> = pilot label, or run1…runK)
    └── experiment<N>/
        ├── cognitive_models/                models going into this experiment → survivors after it
        ├── design/stimuli.json              the 64 pairs and their information gain; screened_out.json
        ├── experiment/                      the agent-built index.html, config.json (study id, URLs)
        ├── deployment/deployment_manifest.json   everything about the deploy and the Prolific study
        ├── data/responses.csv               the human responses: CONTAINS PROLIFIC IDs
        ├── data/observability.log           collection log
        ├── model_loop/                      fits, proposals, critiques, report.md, history.json
        ├── model_registry.yaml              equal-weight prior for the next design
        └── token_usage_summary.json         language-model spending
```

On Firebase, each participant's data sit in Firestore under
`collection_sessions/<collection_session_id>/responses`. They are reachable
only through the token-protected `/results` function or with Google Cloud
credentials.

**`$SCRATCH` is purged**: files not modified for 90 days are deleted. Move
anything you need to keep. Raw data with IDs go to access-controlled storage
such as `$OAK` or `$GROUP_HOME`, as your IRB protocol allows; do not refresh
files to dodge the purge. Browse a run with
`uv run python -m src.viewer.server --data-root $WORK_ROOT/<label>/data`. This
is a web server, so run it inside a job (`sh_dev`) and reach it through an SSH
port forward, not on a login node.

## 10. Recovering a stalled or crashed run without paying again

`run_live.sbatch` accepts `RESUME_AGENTS`: stages separated by `:`, e.g.
`4_collect:5_model_loop`. With it, the job re-runs **only** those stages of an
existing experiment. The deploy is tied to `3_implement`, so leaving that out
never redeploys or recruits.

Example: modelling experiment 1 of the pilot `pilot4` again after problem A is fixed:

```bash
export REPO=$HOME/auto-psych OUTER_LIVE_SLURM_DIR=$HOME/auto-psych/scripts/outer_loop_live
WORK_ROOT=$SCRATCH/auto-psych/outer_loop_live; LABEL=pilot4
# The run copy is a snapshot: copy the fixed code into it first (same rsync as in R2,
# with WT=$WORK_ROOT/runs/$LABEL/repo)
sbatch --job-name=resume_$LABEL --time=12:00:00 \
  --output=$WORK_ROOT/slurm_logs/%x_%j.out --error=$WORK_ROOT/slurm_logs/%x_%j.out \
  --export=ALL,RUN_LABEL=$LABEL,RUN_WORKTREE=$WORK_ROOT/runs/$LABEL/repo,AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/$LABEL/data,EXPERIMENT=1,N_PARTICIPANTS=10,PROLIFIC_MODE=none,RESUME_AGENTS=5_model_loop,DRAWS=2000,TUNE=2000,CHAINS=4,INNER_LOOP_ITERATIONS=2,INNER_LOOP_CANDIDATES=3 \
  $OUTER_LIVE_SLURM_DIR/run_live.sbatch
```

- I did not check how the model stage treats the half-finished `model_loop/`
  left by the crashed attempt. The conservative choice is to move it aside
  first: `mv …/experiment1/model_loop …/experiment1/model_loop.failed`.
  `cognitive_models/` is only rewritten at the very end of the stage, so a
  crash leaves it as it was.
- `PROLIFIC_MODE=none` is safe here. It does not need the live confirmation
  flag, and live collection still works: `--mode live` reads the study id from
  `experiment/config.json`.
- Re-running `4_collect` waits for the study's target again (immediately
  satisfied if it was already reached) and downloads everything again.
- To go on to the next experiment afterwards, launch that experiment
  alone with `EXPERIMENT=2` (all stages, new study). Never relaunch the
  whole range.
- `N_PARTICIPANTS` sets the design's N in a new experiment, so keep it equal
  to the config's `participants`.

## 11. Collecting and storing the results

`scripts/outer_loop_live/collect_results.sh` wraps
`collect_human_results.py`. It copies the finished runs' result trees from
`$WORK_ROOT` into `$REPO/data/results/human_experiment/` and **removes every
Prolific ID** on the way:

- it drops the `participant_id_str` column (and other ID-like columns) from
  every CSV;
- it redacts every 24-character hexadecimal token (Prolific's ID format) in
  every text file: logs, manifests, transcripts;
- it re-scans the copy and **fails** if any ID survived.

It leaves out the MCMC trace files (`.nc`), `node_modules` and the run copies,
and writes a `SUMMARY.md` with the best model, posterior and ELPD margin per
run and experiment.

```bash
export REPO=$HOME/auto-psych; cd $REPO
# defaults: SOURCE=$SCRATCH/auto-psych/outer_loop_live, RUNS="run1 run2 run3",
#           DEST=data/results/human_experiment, PY=$REPO/.venv/bin/python
PY=$SCRATCH/auto-psych/outer_loop_live/venv/bin/python RUNS="run1 run2 run3" \
DEST=data/results/human_experiment_<month> \
  bash scripts/outer_loop_live/collect_results.sh            # runs itself in a short 'dev' job
# a pilot:  RUNS="pilot4" DEST=data/results/pilot4 ... collect_results.sh
```

- **Set `DEST` to a new directory.** The default,
  `data/results/human_experiment/`, already holds the scrubbed results of the
  earlier human study (three runs × three experiments). The collector refuses
  a non-empty destination, and `--overwrite` would **replace** those results.

- The default `PY` is a `.venv` inside the checkout (made by `uv sync`), which
  may not exist on the cluster. Point `PY` at any environment with `tyro` and
  `pyprojroot`, such as the live venv above.
- Pilots (`pilot*`) and `_validate*` runs are skipped by auto-discovery. Naming
  them in `RUNS` includes them.

Only this scrubbed copy may be committed or shared. Keep the raw `$WORK_ROOT`
trees, and any copy of them, in access-controlled storage per your IRB
protocol.

## 12. Privacy rule for Prolific IDs

A Prolific ID (24 hexadecimal characters) identifies a person on Prolific. Treat
it as identifying data:

- **Never commit, publish, paste or send** a file that contains one:
  `data/responses.csv`, `experiment/config.json`, `deployment_manifest.json`,
  logs, the dashboard. That includes sending it to a chatbot or coding agent.
  Share and commit only the output of the collector in § 11.
- The live dashboard shows IDs. Keep it on `127.0.0.1`; `--host 0.0.0.0` prints
  a warning for this reason.
- `python -m src.viewer.freeze` (the public results snapshot) copies response
  previews and agent transcripts verbatim. Freeze only scrubbed runs, and read
  what it lists before deploying it.
- Until problem A is fixed, the IDs also sit in `model_loop/responses.csv`,
  which the agents read. See § 0.
- Sherlock is approved for Low and Moderate Risk data only. If you are unsure
  how your IRB classifies these data, ask before collecting.

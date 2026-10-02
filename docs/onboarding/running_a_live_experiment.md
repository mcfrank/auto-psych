# Running a live experiment (Prolific + Firebase) on Sherlock

Every command here was checked against the scripts and command-line
interfaces on 28 September 2026. On 29 September 2026 R1 and R3 (§ 5) were
run for real and passed, including a real Firebase deploy and a test
submission read back through `/results`. R2 and R4 were run and stopped at
the page-building agent, because the Gemini key was on the free tier (§ 1).
No study has been published from these instructions yet. Placeholders:
`$REPO` is your checkout, `$WORK_ROOT` is where live runs write (default
`$SCRATCH/auto-psych/outer_loop_live`), `<label>` is a run label you choose.

**The order of a full study**, each step below: credentials (§ 1), setup
(§ 2), config (§ 3), the four rehearsals (§ 5), launch (§ 6), monitoring
(§ 7), then collecting and checking the data (§ 10). For the full run
(three runs × three experiments) with Claude agents and its
results committed to `main`, follow [full_run_checklist.md](full_run_checklist.md).

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
| `PROLIFIC_API_TOKEN` | creating, publishing, polling and pausing studies (Prolific → Settings → API tokens) |
| `FIREBASE_TOKEN` | `firebase deploy` from a compute node (make it with `npx -y firebase-tools login:ci` on a machine with a browser) |
| `AUTO_PSYCH_RESULTS_TOKEN` | shared secret protecting `/results`. Reuse the value any checkout already has (`grep -l '^AUTO_PSYCH_RESULTS_TOKEN=' $HOME/repos/*/.secrets`): every deploy writes it into the project-wide functions, so a new value locks out earlier deployments' collection. Only if none exists: `openssl rand -hex 32` |
| `GOOGLE_API_KEY` | the Gemini login of the default agents (opencode). **It must belong to a Google Cloud project with billing enabled.** A free-tier key cannot run the agents: `gemini-3.1-pro` has a free quota of 0, and `gemini-3.7-flash` allows 5 requests a minute and 20 a day, which one page-building attempt uses up (seen 29 September 2026: the agent quits without writing `index.html`) |

With `coding_agent: claude`, state how the agents are billed with
`claude_auth` in the config: `subscription` needs `CLAUDE_CODE_OAUTH_TOKEN`
(from `claude setup-token`), `api` needs `ANTHROPIC_API_KEY`. There is no
default; the run stops before any agent starts without the mode or its key,
and each agent gets only that one credential. The agents never see the
Prolific, Firebase or results tokens.

## 2. One-time setup

Sherlock's git is old: `git -C`, `git remote get-url` and `git worktree` do not
exist. Use `cd` in a sub-shell, `git config --get remote.origin.url`, and
separate clones. If another session runs from your main checkout, make a
second clone for live runs and put its `REPO` and `OUTER_LIVE_SLURM_DIR`
exports in a small file you `source` in each new shell (not `~/.bashrc`).

```bash
export REPO=$HOME/auto-psych                      # your checkout: ALWAYS export this (_env.sh defaults to $HOME/auto-psych)
export OUTER_LIVE_SLURM_DIR=$REPO/scripts/outer_loop_live
sbatch $OUTER_LIVE_SLURM_DIR/setup.sbatch          # 40 min, 4 CPUs; builds $WORK_ROOT/venv
```

The log (`outer_live_setup_<jobid>.out`, in the directory you submitted from)
must end with `[setup] live import chain OK` and `[setup] done.`. Without
`OUTER_LIVE_SLURM_DIR` the job cannot find `_env.sh`.

`_env.sh` runs `set -euo pipefail`: `source` it only in a sub-shell (type
`bash` first), or a later failing command closes your login shell (in tmux,
the pane). To see why a command fails without losing the pane, run it as
`bash -c 'source $OUTER_LIVE_SLURM_DIR/_env.sh; …' 2>&1 | tee some.log`.

`_env.sh` also writes `$WORK_ROOT/bin/node`, a wrapper that puts the modules'
libraries back for the child `node` that `firebase deploy` starts. Without
it the functions deploy fails on el7 but still exits 0, so every Firebase
command must run in a shell or job that sourced `_env.sh`.

**Do not run `uv run` or `uv sync` in a shell where `UV_PROJECT_ENVIRONMENT`
points at `$WORK_ROOT/venv`** (it does after sourcing `_env.sh`, and may in
your login profile). `uv` then re-syncs the live venv to the project lock and
installs arviz 1.x / pymc 6, which break the model stage. Use `"$VENV_PY"`
for the live pipeline. For the test suite, use a separate environment, for
example in a dev job: `UV_PROJECT_ENVIRONMENT=$L_SCRATCH/test_venv uv sync --locked`
then `$L_SCRATCH/test_venv/bin/python -m pytest -q`. To check the live venv:
`"$VENV_PY" -c "import arviz, pymc; print(arviz.__version__, pymc.__version__)"`
must print `0.23.x 5.28.5`; if not, rerun `setup.sbatch`.

## 3. The config

Copy a preset, never edit it in place: `cp $REPO/scripts/outer_loop_live/pilot.yaml $REPO/my_pilot.yaml`.

| preset | launcher | settings |
|---|---|---|
| `pilot.yaml` | `run_pilot.sh` (one run) | 2 experiments × 10 people; 2 rounds × 3 proposals; 2,000 draws |
| `full_run.yaml` | `start_full_run.sh` (K parallel runs, default 3) | 3 × 40; 5 × 6, as the simulations (only the draws differ: 3,000); **use this one** |
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
| `agent_model` | optional: the agents' model, e.g. `google/gemini-3.7-flash` (exported as `CODING_AGENT_MODEL`). Default: `google/gemini-3.1-pro-preview` for opencode, `claude-sonnet-4-6` for claude. Printed in the launcher summary |
| `prolific_mode` | `test` (the pilot preset): deploy, create a draft study, not published, then stop. `live`: publish, recruit, pay, model. `none`: deploy the page, no study, then stop. Missing key ⇒ `test`. |
| `confirm_live_recruitment` | must be `true` for `live`. `full_run.yaml` and `hero_run.yaml` say `live` without it, so they are refused until you choose. |
| `walltime`, `qos` | Slurm limit; `qos: long` above 2 days, and only there: it refuses a limit under 48 hours |
| `prolific.participants` | per experiment: the places recruited, the collection target **and** the design's N |
| `prolific.reward_per_hour` (cents) or `reward` (cents flat), `estimated_completion_time` (min) | pay |
| `prolific.name`, `description`, `completion_code`, `min_approval_rate` | study settings; use a distinct completion code per series |
| `prolific.completion_code_action` | `AUTOMATICALLY_APPROVE` (default) or `MANUALLY_REVIEW` |
| `prolific.exclude_earlier_participants` | `true` (default): a live study excludes everyone who took part in an earlier published study of the pipeline in the Prolific account (every study it named `auto-psych …`, read from the account just before the study is created; Prolific applies the list when the study is published, so people still taking a parallel run's study at that moment are not excluded). `false` only to recruit the same people on purpose |
| `modeling.inner_loop_iterations`, `inner_loop_candidates`, `draws`, `tune`, `chains` | rounds, proposals per round, MCMC (the job has 16 CPUs and 128 GB: four 4-chain fits at once; 64 GB ran out in experiment 3's model loop, where the loop holds about 1.5 GB per model) |
| `modeling.target_accept`, `agent_timeout_sec` | NUTS target acceptance (unset: the model's own, else 0.99) and seconds per critique/proposal agent (unset: 900); `full_run.yaml` sets 0.8 and 1800, as the simulations |
| `modeling.novelty_rmse_threshold`, `prune_dse_multiplier`, `candidate_parallelism`, `hints_file` | optional; defaults 0.002, 2.0, all at once, the built-in eleven angles |

**Cost** (printed by the launcher): pay per person = cents/hour × minutes / 60,
plus an estimated 33% Prolific fee (hard-coded; check your account). At
$12/h and 7 minutes: $1.40 a person; a pilot (2 × 10) ≈ $37; a full or hero
run (3 × 40) ≈ $223 per run, ≈ $670 for K = 3. Language-model costs come on
top and are not estimated; each experiment records its spend in
`token_usage_summary.json`. Prolific takes each study's whole cost (pay and
fee) from the workspace's **available** balance when it is published, and a
publish it cannot cover fails, so before a launch check that the available
balance (Prolific dashboard, or `GET /workspaces/<id>/balance/`) covers every
study the launch will publish; money held for submissions awaiting review is
not available.

**Time.** Per experiment: a few minutes of design, up to three 15-minute
attempts at the page, the deploy, **up to 3 hours of recruiting**, then the
model stage (with `full_run.yaml`'s 5 × 6 rounds, roughly 5–8 hours an
experiment, judging by the simulations; a proposal's fit may take up to 30
minutes, and agents that hit a usage limit wait for it to reset). Three
experiments come to roughly a day or a day and a half; `full_run.yaml`'s
`walltime` is 47 hours.

## 4. Safety gates

| gate | what it stops |
|---|---|
| `confirm_live_recruitment: true` (launcher) and `--confirm-live-recruitment` (`run.py`) | one edited word spending money |
| cost summary, config check, Prolific token check, typed `yes` | typos and accidental launches (`CONFIRM=yes` skips the prompt: not for live runs) |
| preflight in `run_pilot.sh`: venv, `FIREBASE_TOKEN`, `PROLIFIC_API_TOKEN`, consent text; `_env.sh` stops without `bwrap` | failing hours into a job (the results token is checked later, before anything is deployed) |
| `--n-participants` is the only count; a rendered `prolific_config.yaml` with a different `total_available_places`, or none at all, stops `run.py` | recruiting a different number than the design assumed |
| results token (and, for `live`, Prolific's eligibility settings) checked first; the page and functions are deployed, the page checked live and `/results` checked to refuse a read without the token (403) and accept one with it (200) **before** the draft study is created, recorded and (live only) published | recruiting onto a broken or unprotected page; functions that were never replaced; a failed deploy leaving a study behind |
| the deploy records the commit the code came from (from git, or from the record the launcher writes into the run copy) and refuses without one | a study that cannot be traced to its code |
| relaunch guard: an experiment whose `deployment/deployment_manifest.json` records a live study refuses `2_design`, `3_implement` and the deploy (`LiveStudyAlreadyRecorded`) | a second paid study for the same experiment |
| a live study excludes the participants of every earlier published pipeline study in the account; the eligibility check requires Prolific's `previous_studies_blocklist` filter, and a study is never created when the account's studies cannot be listed | the same person taking the task twice (two did in the October 2026 series' run 1) |
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

**Environment check** (in the same sub-shell; prints no secret values):

```bash
which node                                 # must be $WORK_ROOT/bin/node (the wrapper)
"$VENV_PY" -c "import arviz, pymc; print(arviz.__version__, pymc.__version__)"   # 0.23.x 5.28.5
for k in FIREBASE_TOKEN AUTO_PSYCH_RESULTS_TOKEN PROLIFIC_API_TOKEN GOOGLE_API_KEY; do
  [[ -n "${!k:-}" ]] && echo "$k set" || echo "$k MISSING"; done
(cd $REPO && git log --oneline -1)        # the code you are about to deploy
```

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

This always rehearses opencode with its default model. To rehearse your
config's agents, add `CODING_AGENT=claude,CLAUDE_AUTH=<subscription|api>` or
`CODING_AGENT_MODEL=<model>` to the `--export` list. Use a new `LABEL` for
each attempt.

(The generated script passes `bash -n`.) It exercises design, the
agent-built page (built, not deployed), collection, the whole model stage,
the carry-over and experiment 2's design. Success: the log contains
`All experiments complete.`, and both experiments have `model_loop/report.md`
and `cognitive_models/models_manifest.yaml`. Open
`experiment1/experiment/index.html` in a browser.

**Is the page-building agent working?** (R2 and R4; at most once a minute.)
The design step comes first and has no agent: `design/stimuli.json` appears
within minutes, and `design/screened_out.json` should be `[]`. Then:

```bash
D=$WORK_ROOT/<label>/data/subjective_randomness/experiment1
tail -n 20 <job log>
ls -la $D/experiment                       # index.html appears when the agent succeeds
grep -o 'error.error="[^"]\{0,100\}' $D/logs/.xdg_data/opencode/log/opencode.log | sort | uniq -c
```

Healthy: `[step] tokens=… cost=$…` lines in the job log, then
`[agent] 3_implement completed.` with no `[repair] … index.html not found`.
Broken: `opencode.log` fills with `exceeded your current quota` (your key's
plan: § 1) or `high demand` (Google's side, temporary), and the job log
shows `finished without success`. The agent's `3_implement.jsonl` is
overwritten by each attempt; `opencode.log` keeps all of them.

**R3. Real Firebase deploy, no study.** Needs `FIREBASE_TOKEN` and
`AUTO_PSYCH_RESULTS_TOKEN`; click through the real page, consent included.

```bash
AUTO_PSYCH_OUTPUT_DIR=$WORK_ROOT/rehearsal-fb/data \
srun -p dev -t 15:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only \
  --deploy-target firebase --prolific-mode none \
  --firebase-project auto-psych-2c5da --run-label rehearsal-fb
```

Pass: the log shows `[deploy] Functions live: /results refuses reads without
the token` and ends `All experiments complete.` Then check from outside the
pipeline. `curl` on Sherlock cannot do it: el7's `curl` fails the TLS
handshake (`curl: (35) … same issuer/serial`, printed as `HTTP 000`), so use
Python. This posts one clearly labelled test row (`rehearsal-check`) to the
rehearsal's own session and reads it back:

```bash
M=$WORK_ROOT/rehearsal-fb/data/subjective_randomness/experiment1/deployment/deployment_manifest.json
"$VENV_PY" - "$M" <<'PY'
import json, os, sys, urllib.error, urllib.parse, urllib.request
m = json.load(open(sys.argv[1])); host = "https://auto-psych-2c5da.web.app"
def call(method, path, body=None, token=None):
    h = {"Content-Type": "application/json", "User-Agent": "auto-psych"}
    if token: h["x-results-token"] = token
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(urllib.request.Request(host + path, data=data, method=method, headers=h), timeout=30) as r:
            return r.status, r.read().decode()[:300]
    except urllib.error.HTTPError as e:
        return e.code, ""
q = "/results?" + urllib.parse.urlencode({"collection_session_id": m["collection_session_id"]})
print("page", urllib.request.urlopen(urllib.request.Request(m["experiment_url"], headers={"User-Agent": "x"}), timeout=30).status)  # 200
print("results, no token", call("GET", q)[0])                                                                                # 403
print("submit", call("POST", "/submit", {"collection_session_id": m["collection_session_id"], "participant_id": "rehearsal-check",
      "trials": [{"sequence_a": "HTHT", "sequence_b": "HHHH", "chose_left": True}]})[0])                                    # 200
print(*call("GET", q, token=os.environ["AUTO_PSYCH_RESULTS_TOKEN"]), sep="\n")                                              # 200 + the row
PY
npx -y firebase-tools functions:list --project auto-psych-2c5da   # only results and submit
```

Then open `https://auto-psych-2c5da.web.app/e1-rehearsal-fb/` in a browser
and click through the consent page and a few trials. The site's root URL
shows Firebase's "Page Not Found": pages live only under `/e<N>-<label>/`.

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

The draft is created **last**, after the agent has built the page (up to
three 15-minute attempts) and the deploy has passed, so for the first half
hour or more there is nothing in Prolific. Where to look:

1. The job log: design, then `3_implement`, then `[deploy] Functions live`.
   If the agent fails (see "Is the page-building agent working?" above),
   no draft is ever created.
2. `$WORK_ROOT/<label>/data/subjective_randomness/experiment1/deployment/deployment_manifest.json`
   gets a `prolific_study_id` the moment the draft exists.
3. Prolific → **Studies**, in the **Unpublished** section.

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
(`https://auto-psych-2c5da-<run_label>-run<i>.web.app/e<N>-run<i>/`, named
after the config's `run_label` so a new series never reuses an earlier
series' sites; the launcher lists them before `yes` and stops on a name over
30 characters), studies, and log
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
- The run copy is a snapshot: to use fixed code, rsync it in first (as in R2),
  into the run's own copy `$WORK_ROOT/runs/$LABEL/repo`, never a new one:
  Firebase serves the copy's whole `public/`, which holds the earlier
  experiments' pages (the rsync leaves it alone), so a deploy from a fresh
  copy takes them offline. Rsync only after the run's job has ended: its fit
  processes import from the copy.
  The rsync replaces the study settings `run_pilot.sh` rendered into the copy
  and deletes its commit record. Render the settings again with
  `"$VENV_PY" $WORK_ROOT/runs/$LABEL/repo/scripts/outer_loop_live/_pilot_config.py <your.yaml> --render-only`
  before any run with a Prolific mode other than `none`, and, before any
  deploy, record the commit again from your checkout:
  `cd $REPO && "$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout $REPO --copy $WORK_ROOT/runs/$LABEL/repo`.
- To run the remaining experiments, submit the same command without
  `RESUME_AGENTS`, with `EXPERIMENTS=<next>-<last>`, `PROLIFIC_MODE=live`
  and `CONFIRM_LIVE_RECRUITMENT=1`; for a parallel run `run<i>` also
  `AUTO_PSYCH_HOSTING_SITE=auto-psych-2c5da-<run_label>-run<i>`, the site
  the launch printed (without it the deploy goes to the default site). This publishes new studies and pays.
  Both can be queued at once: submit the resumed stage first, then the
  remaining experiments with `--dependency=afterok:<its job id>`, so they
  start only if the stage succeeds (if it fails, the second job stays
  pending with `DependencyNeverSatisfied`: `scancel` it). `--qos=long`
  refuses a time limit under 48 hours; leave it out below that. To queue the
  remaining experiments on code the running stage's copy does not have yet,
  rsync the checkout into a second directory beside the copy now (and record
  its commit there), and make the queued job rsync that directory into the
  copy when it starts, with the same excludes: the copy cannot change while
  the running job imports from it.
- A second study for the same experiment: stop the first in Prolific, then
  set `PUBLISH_ANOTHER_PROLIFIC_STUDY=1`; the old manifest is kept as
  `deployment_manifest.superseded-<time>.json`.

## 10. Where the data are, collecting them, and checking them

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

**Checking the data.** `experiment<N>/data/responses.csv` is what the models
see: the five columns `sequence_a, sequence_b, participant_id, trial_index,
chose_left`, and no Prolific IDs. For each experiment:

```bash
"$VENV_PY" - $WORK_ROOT/<label>/data/subjective_randomness/experiment1/data/responses.csv <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
print("columns", list(d.columns))
print("participants", d.participant_id.nunique(), "rows", len(d))
per = d.groupby("participant_id").agg(trials=("trial_index", "size"), left=("chose_left", "mean"))
print(per.describe().round(2))
print("one-sided participants (>= 95% one side):", int(((per.left <= 0.05) | (per.left >= 0.95)).sum()))
PY
```

Expect as many participants as `prolific.participants` (fewer if the 3-hour
wait ended short; the study was then paused), 64 trials each, and a left
rate near 0.5 overall. Compare the count with Prolific's submissions.
Even with `AUTOMATICALLY_APPROVE`, Prolific holds back a submission faster
than its threshold (in the October 2026 series, everyone under about 3.5
minutes of the 7 estimated: 31 of run 1's 120) as `AWAITING REVIEW`, and
the participant is not paid until you approve or reject it in the dashboard;
the submission's `time_taken_under_auto_approval_threshold` says which.
Review those (their data are in `responses.csv` like everyone's).
`raw_collected/experiment<N>_responses.csv` has the same rows
with Prolific IDs (look at it only on Sherlock). A participant answering one
side on every trial is the failure the live dashboard (§ 7) is there to
catch early. Then read `experiment<N>/model_loop/report.md` for the model
stage's result.

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

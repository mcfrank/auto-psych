# The full live run, start to finish: a checklist

Written 29 September 2026 for the next full human run: **3 independent runs
of the whole pipeline × 3 experiments each × 40 participants = 9 Prolific
studies, 360 people**, with Claude Opus 5.5 agents, in the Firebase project `auto-psych-2c5da`,
launched from `main`, with results committed to `main` for collaborators. Each step points to the
section of [running_a_live_experiment.md](running_a_live_experiment.md) (the
runbook, "§") that explains it; this page only fixes the order and the
settings for this run.

Not yet run as written: Claude agents in a live run and the results sync
(steps 5, 10–12). The rehearsal in step 5 is there to find out.

Placeholders: `<name>` names this series of runs (lowercase letters, digits
and `-`); it is the config's `run_label`, which names the runs' Hosting
sites. `$REPO` is the live checkout and `$WORK_ROOT` is
`$SCRATCH/auto-psych/outer_loop_live` (the default), as in the runbook.

## Before: accounts, project, config

1. **Firebase: the existing project `auto-psych-2c5da`** (user decision,
   29 September 2026). Its Firestore also holds the earlier human run and
   March 2026 test data; each run's data are kept apart by their collection
   session (`/results` is always read by session). Each run deploys to a
   Hosting site of its own, **`auto-psych-2c5da-<run_label>-run<i>`**
   (`scripts/outer_loop_live/_hosting_site.sh`), so this series never reuses
   the earlier series' `auto-psych-2c5da-run<i>` sites. Site IDs are at most
   30 characters, so `run_label` can be **at most 8** (`auto-psych-2c5da-`
   and `-run<i>` take 22); the launcher refuses a longer one before its
   prompt. The deploy creates a site that does not exist yet. All runs share
   the project's `/submit` and `/results` functions, so every run of a
   series must deploy the same code (step 7). Keep `AUTO_PSYCH_RESULTS_TOKEN`
   unchanged: every deploy writes it into those functions (§ 1).

2. **Credentials** in `$REPO/.secrets` (§ 1): `PROLIFIC_API_TOKEN`,
   `FIREBASE_TOKEN`, `AUTO_PSYCH_RESULTS_TOKEN`, and the Claude login for the
   billing mode you choose: `ANTHROPIC_API_KEY` for `api`,
   `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) for `subscription`.
   Prefer `api`: nine runs of up to six concurrent Opus agents will reach a
   subscription's session limit, and every wait for a reset (up to 12 hours
   each) comes out of the job's walltime.

3. **The config.** Copy the preset, never edit it in place (§ 3):

   ```bash
   cp $REPO/scripts/outer_loop_live/full_run.yaml $REPO/full_run_<name>.yaml
   ```

   and change only these keys (keep the `modeling:` block, which
   `tests/test_live_run_settings.py` keeps equal to the simulated checks):

   ```yaml
   run_label: <name>                  # names the sites; at most 8 characters (step 1)
   firebase_project: auto-psych-2c5da
   coding_agent: claude
   claude_auth: api                   # or subscription (step 2)
   agent_model: claude-opus-5-5       # the claude default is claude-sonnet-4-6
   prolific_mode: live
   confirm_live_recruitment: true
   walltime: "3-00:00:00"             # see below
   qos: long                          # needed above 2 days
   prolific:
     completion_code: AUTO_PSYCH_COMPLETE_<NAME>   # distinct per series
   ```

   The runs themselves are `run1`, `run2`, `run3` (their directories, page
   paths `/e<N>-run<i>/` and sessions).
   **Walltime:** `full_run.yaml`'s 47 hours was budgeted for opencode agents;
   three experiments of up to 3 hours' recruiting plus 5–8 hours of
   modelling each come to roughly 25–35 hours before any usage-limit waits.
   Check the partition's limits with `sh_part`.

4. **The cost** the launcher prints: at $12/hour and 7 minutes, about
   $223 per run, **about $670 for the three**, plus Opus agent costs, which
   the launcher does not estimate (step 5's rehearsals measure them in
   `token_usage_summary.json`).

## Rehearse with these agents (§ 5)

5. The rehearsals so far used opencode agents, so rerun **R2** (simulated
   participants, no Firebase, no Prolific) with `coding_agent: claude`,
   `claude_auth` and `agent_model: claude-opus-5-5`: it shows the Claude
   agents run in the sandbox on compute nodes, and what an experiment
   costs. Use a new label. The deploy itself passed R3 and R4 on
   29 September 2026 in this project.

## Launch and watch

6. **Launch from `main`, with everything committed.** Commit your config
   to `main` (it holds no secrets, and collaborators then have it too):

   ```bash
   cd $REPO && git checkout main && git pull
   git add full_run_<name>.yaml && git commit -m "Full run <name>: config" && git push
   git status --short                      # must print nothing
   ```

   An empty `git status` matters: the launcher records the commit in every
   deployment manifest, and any modified *or untracked* file marks the run
   `git_dirty: true`.

7. **Launch** (§ 6), from a login node; you type `yes` yourself:

   ```bash
   source $HOME/repos/live_env.sh
   AUTO_PSYCH_COLLECTION_OWNER=<you> CONFIG=$REPO/full_run_<name>.yaml \
     bash $REPO/scripts/outer_loop_live/start_full_run.sh
   ```

   `AUTO_PSYCH_COLLECTION_OWNER` labels every manifest and Firestore record
   (otherwise `unknown`). To start one run first and the other two later,
   prefix `RUNS=1`, then `RUNS="2 3"`: each launch copies the checkout as it
   is then, so launch the later runs from the **same commit** as run 1
   (compare `git rev-parse HEAD` with the `git_commit` in run 1's
   `deployment_manifest.json`); all runs share the project's functions.

   Before typing `yes`, check the summary: the `LIVE` banner, 40
   participants, `prolific token : OK`, the agent model
   `claude-opus-5-5`, the `hosting sites` line, and **the list of directories it will delete**
   (`$WORK_ROOT/run<i>`, `$WORK_ROOT/runs/run<i>`). It must say it deletes
   nothing; if it lists anything, collect those results first (step 11).
   Each run's pages are at
   `https://auto-psych-2c5da-<name>-run<i>.web.app/e<N>-run<i>/`.

8. **Watch** (§ 7): `squeue --me`, the job logs
   (`$WORK_ROOT/slurm_logs/outer_live_run<i>_<jobid>.out`), each
   experiment's `data/observability.log`, the Prolific dashboard and the
   live dashboard (§ 7). About once a minute at most, never in a loop.

9. **When something breaks**: pause the studies in Prolific first, then
   `scancel` (§ 8); recover with `RESUME_AGENTS`, never by relaunching an
   experiment that has a study (§ 9). For a parallel run the resume needs
   `AUTO_PSYCH_HOSTING_SITE=auto-psych-2c5da-<name>-run<i>` (the site the
   launch printed).

## Share the results: commits to `main`

The results are committed straight to `main`, in the same checkout, as
scrubbed copies under `data/results/human_experiment_<name>/`. The running
jobs use their own copies of the code (`$WORK_ROOT/runs/run<i>/repo`), so
committing in the checkout never disturbs them.

10. **Once, at launch:** add `data/results/human_experiment_<name>/README.md`
    for collaborators (template below), commit it and push to `main`.

11. **After each experiment finishes** (every run's
    `experiment<N>/model_loop/export_complete.json` exists), sync:

    ```bash
    cd $REPO && git pull
    D=data/results/human_experiment_<name>/collected
    rm -rf $D
    PY=$SCRATCH/auto-psych/outer_loop_live/venv/bin/python RUNS="run1 run2 run3" DEST=$D \
      bash scripts/outer_loop_live/collect_results.sh     # a short dev job (§ 10)
    grep -rlwE '[0-9a-f]{24}' $D && echo "STOP: Prolific IDs above" || echo "no Prolific IDs"
    git add -A $D && git commit -m "Full run <name>: sync after experiment <N>" && git push
    ```

    Only if the `grep` prints `no Prolific IDs`, commit. `collected/` is
    regenerated whole each time: the collector copies over an existing
    destination without removing files that have since disappeared from a
    run (a model-loop restart empties `model_loop/`). It removes every
    Prolific ID and fails if one survives (§ 10, § 11); the `grep` (the
    collector's own pattern) is a second check before anything is pushed.
    The next experiment's partial directory is copied too; `SUMMARY.md`
    marks it `incomplete`. Never `uv run` or `uv sync` here (§ 2).

12. **At the end**: a last sync, then copy
    `$WORK_ROOT/run<i>/data/subjective_randomness/raw_collected/` (it holds
    Prolific IDs) to access-controlled long-term storage your IRB protocol
    allows (§ 10). `$SCRATCH` is purged after 90 days without modification.

### README template for the results directory

```markdown
# Full live run <name>

Three independent runs (run1–run3) of the whole pipeline, three experiments
each, 40 Prolific participants per experiment. Code: the commit recorded as
`git_commit` in each experiment's `deployment/deployment_manifest.json`.
Config: `full_run_<name>.yaml` at the repository root. Agents: Claude Opus 5.5.

An experiment is final when its `model_loop/export_complete.json` exists;
`collected/SUMMARY.md` lists each experiment's winning model and marks unfinished ones
`incomplete`.

Per run, `collected/run<i>/subjective_randomness/experiment<N>/` holds:
- `data/responses.csv`: the models' data (`sequence_a, sequence_b,
  participant_id, trial_index, chose_left`); `participant_id` is numbered
  within the run and never a Prolific ID
- `design/stimuli.json`: the chosen stimulus pairs and their expected information
- `cognitive_models/`: the models carried to the next experiment, with
  `models_manifest.yaml` and the ledger of every hypothesis tried
- `model_loop/`: every round's proposals, critiques, fits, `history.json`,
  `model_posterior.json` and `report.md`

Browse it: `python -m src.viewer.server` (see `src/viewer/`).
```

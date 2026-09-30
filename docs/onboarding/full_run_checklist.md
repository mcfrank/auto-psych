# The full live run, start to finish: a checklist

Written 29 September 2026 for the next full human run: **3 independent runs
of the whole pipeline × 3 experiments each × 40 participants = 9 Prolific
studies, 360 people**, with Claude Opus 5.5 agents, launched from `main`, with results
committed to `main` for collaborators. Each step points to the
section of [running_a_live_experiment.md](running_a_live_experiment.md) (the
runbook, "§") that explains it; this page only fixes the order and the
settings for this run.

Not yet run as written: a new Firebase project, Claude agents in a live run
and the results sync (steps 1, 3, 10–12). The rehearsals in step 5 are
there to find out.

Placeholders: `<name>` names this series of runs (lowercase letters, digits
and `-`; for October 2026, `sr-oct26`), `<fb>` is its Firebase project ID
(for October 2026, `autopsych-sr-oct26`). `$REPO` is the live checkout and
`$WORK_ROOT` is `$SCRATCH/auto-psych/outer_loop_live` (the default), as in
the runbook.

## Before: accounts, project, config

1. **A Firebase project for this series.** One project holds one Firestore
   database, the `/submit` and `/results` functions and the Hosting sites, so
   a new project keeps this series' data apart from earlier runs and tests
   (`auto-psych-2c5da` holds the earlier human run and March 2026 test
   data). In the [Firebase console](https://console.firebase.google.com):
   - Create the project, choosing its ID (`<fb>`, e.g. `autopsych-<name>`;
     if Firebase says the ID is taken and appends a suffix, use the ID it
     shows). **At most 25 characters**: each run deploys to its own Hosting site
     `<fb>-run<i>`, and site IDs are capped at 30. The ID replaces
     `auto-psych-2c5da` everywhere below, including the runbook.
   - Upgrade it to the **Blaze** plan (Cloud Functions needs it).
   - Create the **Firestore** database (native mode, location
     `us-central1`, where the functions are deployed).
   - Open **Hosting → Get started** once, which creates the default site
     `<fb>`: the full run's per-run sites `<fb>-run<i>` are created by the
     deploy, but a pilot (step 5's R4) deploys to the default site.
   - Make sure the Google account behind `FIREBASE_TOKEN` is an owner (the
     token from `firebase login:ci`; § 1). Add collaborators who should read
     Firestore under project settings → users and permissions.
   - Keep `AUTO_PSYCH_RESULTS_TOKEN` unchanged: every deploy writes it into
     the project's functions (§ 1).

   Nothing in the pipeline names the project: the config's
   `firebase_project` is passed to every deploy.

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
   firebase_project: <fb>
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

   `run_label` is ignored: the runs are `run1`, `run2`, `run3`.
   **Walltime:** `full_run.yaml`'s 47 hours was budgeted for opencode agents;
   three experiments of up to 3 hours' recruiting plus 5–8 hours of
   modelling each come to roughly 25–35 hours before any usage-limit waits.
   Check the partition's limits with `sh_part`.

4. **The cost** the launcher prints: at $12/hour and 7 minutes, about
   $223 per run, **about $670 for the three**, plus Opus agent costs, which
   the launcher does not estimate (step 5's rehearsals measure them in
   `token_usage_summary.json`).

## Rehearse with this project and these agents (§ 5)

5. The rehearsals so far used `auto-psych-2c5da` and opencode agents, so
   rerun two of them with a copy of **this** config:
   - **R2** (simulated participants, no Prolific) with `coding_agent: claude`
     and `agent_model: claude-opus-5-5`: shows the Claude agents run in the
     sandbox on compute nodes, and what an experiment costs.
   - **R4** (`prolific_mode: test`, a draft study that is never published)
     with `firebase_project: <fb>`: the first deploy to the new project,
     creating its functions and a Hosting site, checked live.

   Use a new `run_label` for each. Delete the R4 draft in Prolific afterwards.

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
   CONFIG=$REPO/full_run_<name>.yaml bash $REPO/scripts/outer_loop_live/start_full_run.sh
   ```

   Before typing `yes`, check the summary: the `LIVE` banner, 40
   participants, `prolific token : OK`, the agent model
   `claude-opus-5-5`, and **the list of directories it will delete**
   (`$WORK_ROOT/run<i>`, `$WORK_ROOT/runs/run<i>`). It must say it deletes
   nothing; if it lists anything, collect those results first (step 11).
   Each run's pages are at `https://<fb>-run<i>.web.app/e<N>-run<i>/`.

8. **Watch** (§ 7): `squeue --me`, the job logs
   (`$WORK_ROOT/slurm_logs/outer_live_run<i>_<jobid>.out`), each
   experiment's `data/observability.log`, the Prolific dashboard and the
   live dashboard (with `gcloud auth application-default set-quota-project
   <fb>`). About once a minute at most, never in a loop.

9. **When something breaks**: pause the studies in Prolific first, then
   `scancel` (§ 8); recover with `RESUME_AGENTS`, never by relaunching an
   experiment that has a study (§ 9). For a parallel run the resume needs
   `AUTO_PSYCH_HOSTING_SITE=<fb>-run<i>` and `FIREBASE_PROJECT=<fb>`.

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

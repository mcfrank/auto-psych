# Troubleshooting

Messages are quoted from the code, shortened with `…`. The code stops with an
error rather than guess, so an error usually needs a decision, not a retry.

**First, for any failure in a live run: look at the Prolific dashboard.** If a
study is published and the job has stopped, pause it, then debug. Recover
with `RESUME_AGENTS` ([runbook § 9](running_a_live_experiment.md#9-recovering-without-paying-again)),
never with a plain relaunch.

| message | cause | what to do |
|---|---|---|
| `PREFLIGHT FAILED: venv not built …` | no `$WORK_ROOT/venv` | runbook § 2 (export `OUTER_LIVE_SLURM_DIR` first) |
| `PREFLIGHT FAILED: FIREBASE_TOKEN missing …` / `PROLIFIC_API_TOKEN missing` | key absent, or `REPO` points at another checkout | add it; `export REPO=<your checkout>` |
| ``pilot config error: `prolific_mode: live` recruits and PAYS … `` | live without `confirm_live_recruitment: true` | add it only if you mean to pay people |
| `…/_env.sh: No such file or directory` (top of a job log) | `sbatch` without `OUTER_LIVE_SLURM_DIR` | export it, resubmit |
| ``RuntimeError: `git rev-parse HEAD` failed in … cannot record deployment provenance`` | the launchers' run copy has no `.git` ([runbook § 0](running_a_live_experiment.md#0-blocking-problem-read-first)) | needs a code change; nothing was created |
| ``live collection needs `prolific_study_id` …`` | `prolific_mode: none` through the launchers, or a fresh experiment resumed with `PROLIFIC_MODE=none` | live collection needs a study |
| `FATAL: bwrap not on PATH …` | the `system bubblewrap` module did not load | `ml spider bubblewrap`; if the module is broken, that is a cluster problem (below) |
| `Error: --prolific-mode live … Pass --confirm-live-recruitment` | `run.py` or `run_live.sbatch` called by hand | `CONFIRM_LIVE_RECRUITMENT=1`, or `PROLIFIC_MODE=none` for a recovery |
| `No Prolific study settings at …` / `… sets total_available_places: …, but this run was started with --n-participants …` | study settings not rendered, or a different count | render with `_pilot_config.py <config> --render-only` in the checkout the job runs from; keep one count |
| `The researchers' raw collected data would be written to …` | `run.py` by hand with the output tree inside the repository | set `AUTO_PSYCH_OUTPUT_DIR` outside it |
| `LiveStudyAlreadyRecorded: experiment<N> already has a live Prolific study …` | relaunch of an experiment with a live study | nothing ran. `RESUME_AGENTS=4_collect:5_model_loop`; a second study only after stopping the first, with `PUBLISH_ANOTHER_PROLIFIC_STUDY=1` |
| `[error] Experiment N's model loop is not complete (…)` | the previous experiment's model stage did not finish | rerun it (`RESUME_AGENTS=5_model_loop` for that experiment) |
| `[error] 3_implement still invalid after 3 attempt(s): …` | the agent could not make a valid page | read `experiment<N>/logs/3_implement.jsonl`; nothing was deployed |
| `AUTO_PSYCH_RESULTS_TOKEN is not set …` | missing key | at deploy time nothing was created; at collection time the study is already running: add it, recover with `RESUME_AGENTS=4_collect:5_model_loop` |
| `Firebase deploy failed …` / `… the experiment page is NOT live …` | expired `FIREBASE_TOKEN`, permissions, outage, hosting not published | nothing was published, but a **draft** study exists and is recorded: delete it in Prolific, fix the cause, then relaunch with `PUBLISH_ANOTHER_PROLIFIC_STUDY=1` (a plain relaunch is refused) |
| `Failed to create Prolific study: …` / `Failed to publish …` | often insufficient funds | a draft may exist: check and delete it |
| log stays at `Prolific poll: … completed=k target=N` | slow recruitment | normal for up to 2 h, then the study is paused |
| `PAUSED Prolific study <id>: collection gave up at k/N …` | 2 hours passed | expected; the partial data are modelled |
| `Could not pause Prolific study …` (or it `… is in state '…'`) | API error or unusual state | **pause it in the dashboard now**, then recover |
| `live results fetch failed …` | wrong token or network | data are safe in Firestore; fix, recover with `RESUME_AGENTS=4_collect:5_model_loop` |
| `Collected data failed the quality check: all N responses are identical …` | broken buttons or bots | inspect the page and the dashboard; nothing was modelled |
| proposal rejected with a reason | a failed admission check | expected, unless every proposal fails every round |
| `"no_critique"` in `history.json` | the critic produced no usable statistic | the round still ran; see `iter_<i>/critique/` |
| `OUT_OF_MEMORY` in `sacct`, or the time limit | large pooled fits, many rounds | raise `--mem` or `walltime`; recover with `RESUME_AGENTS` |

**If the cluster itself looks broken** (scheduler, filesystem, modules,
network): stop, note the symptom, hostname, time and job id, and send them to
srcc-support@stanford.edu yourself. Prolific studies still need pausing by
hand.

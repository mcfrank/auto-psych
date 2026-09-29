# Troubleshooting

The messages below are quoted from the code (shortened with `…`). Each is
grouped by where it appears, with its cause and what to do. The code generally
**fails loudly rather than silently substituting data**, so an error usually
means something needs a decision, not a retry.

First rule for any failure during a live run: **check the Prolific dashboard.**
If a study is published and the pipeline has stopped, pause or stop the study
first, then debug. Never simply relaunch; that deploys and pays again (see
[running_a_live_experiment.md § 0 C and § 10](running_a_live_experiment.md)).

## When launching (`run_pilot.sh`, `start_full_run.sh`)

| message | cause | what to do |
|---|---|---|
| `PREFLIGHT FAILED: venv not built — run once: sbatch …/setup.sbatch` | no `$WORK_ROOT/venv` | build it as in runbook § 2 (export `OUTER_LIVE_SLURM_DIR` first; the command in the message alone fails) |
| `PREFLIGHT FAILED: FIREBASE_TOKEN missing in .secrets` / `PROLIFIC_API_TOKEN missing` | key absent from `$REPO/.secrets`, or `REPO` points at another checkout | add the key; `export REPO=<your checkout>` |
| `PREFLIGHT FAILED: no IRB consent text at templates/consent.txt` | consent file missing | restore the IRB-approved text |
| ``pilot config error: `prolific_mode: live` recruits and PAYS real participants. Add `confirm_live_recruitment: true` …`` | the second live-mode gate | add the line only if you mean to pay people |
| ``pilot config error: `design_mode` was removed …`` | old config key | delete the key |
| ``pilot config error: missing required `prolific.participants` `` (or `project`, `run_label`) | incomplete config | add it |
| `pilot config error: Prolific token check FAILED: GET /users/me/ 401 …` | wrong or expired token | make a new token in Prolific's settings |
| `min_approval_rate must be a percentage between 0 and 100` / `Prolific reward config is non-positive …` | bad `prolific:` values | fix the config |
| `Aborted — nothing was deployed or published.` | you did not type `yes` | — (note that `start_full_run.sh` has already deleted the selected runs' old directories by this point) |

## Early in the job

| message | cause | what to do |
|---|---|---|
| a line ending in `_env.sh: No such file or directory` at the top of a job log | the job was submitted from a directory other than `scripts/outer_loop_live` without `OUTER_LIVE_SLURM_DIR` | `export OUTER_LIVE_SLURM_DIR=$REPO/scripts/outer_loop_live` before `sbatch` |
| `Error: --prolific-mode live recruits and PAYS real participants. Pass --confirm-live-recruitment …` | `run.py`'s own gate, e.g. calling `run_live.sbatch` by hand | export `CONFIRM_LIVE_RECRUITMENT=1`, or use `PROLIFIC_MODE=none` for a recovery (runbook § 10) |
| `Error: experiment directory already exists: … Use --resume …` | calling `run.py` directly into an existing output | new `AUTO_PSYCH_OUTPUT_DIR` or label. Do not add `--resume` to a live run without reading runbook § 0 C. |
| `[error] Model-set validation failed: … (experiment 1 requires project seed models in …)` | starting models missing or unloadable | check `projects/subjective_randomness/seed_models/` in the run copy |
| `Cannot carry the model set forward: … does not exist (did experiment 'experimentN' complete?)` | experiment N+1 launched before N finished | finish or recover experiment N first |
| ``Sandboxed agents need bubblewrap on PATH (on Sherlock: `ml load system bubblewrap`).`` | known gap in `scripts/outer_loop_live/_env.sh` (runbook § 0 B) | add `ml load system bubblewrap` to that file |
| `'opencode' (the opencode CLI) is not on PATH.` (or `'claude'`) | module not loaded | `_env.sh` loads `opencode` and `claude-code`; check `ml spider opencode` |
| `[error] 3_implement still invalid after 3 attempt(s): …` | the agent could not produce a page that passes the validator (e.g. `index.html does not mention jsPsych`) | read `experimentN/logs/3_implement.jsonl`; no deploy has happened yet |

## Deploy

| message | cause | what to do |
|---|---|---|
| `AUTO_PSYCH_RESULTS_TOKEN is not set. Generate a secret …` | results token missing (it is not in `.secrets.example`) | add it to `.secrets`. Nothing was deployed or created. |
| `Firebase deploy requires --firebase-project or a real .firebaserc` | no project id | the launchers always pass one; for direct calls add `--firebase-project auto-psych-2c5da` |
| `IRB consent text not found at …` | consent file missing in the run copy | restore it and relaunch with a new label |
| `Firebase Functions dependencies are missing and npm is not installed` / `Failed to install Firebase Functions dependencies with npm` | Node not loaded or npm failure | `_env.sh` loads Node 24 (firebase-tools does not support Node 25); rerun `setup.sbatch` |
| `Firebase deploy failed (--only functions,firestore)` / `(--only hosting)` + output | expired `FIREBASE_TOKEN`, missing permissions, or a Firebase outage | the output says which. Regenerate the token with `firebase login:ci`. No study has been created yet. |
| `Could not create Firebase Hosting site '…-run<i>'` | parallel-run site creation refused (permissions or quota) | see the output; unverified whether the project has room for more sites |
| `Firebase deploy reported success but the experiment page is NOT live: GET … -> 404 …` | hosting did not publish | no study was published. Investigate before relaunching with a new label. |
| `Could not register collection session …` | `/register_session` rejected the token or was unreachable | results token mismatch between `.secrets` and the deployed functions, or a network problem |
| `Could not fetch Prolific filters …` / `Prolific choice ID drift: …` | Prolific's country/language IDs could not be confirmed | the hard-coded IDs in `deployment/prolific.py` need checking against Prolific. Nothing was created. |
| `Failed to create Prolific study: …` / `Failed to publish Prolific study: …` | Prolific API refused (often insufficient funds or invalid fields) | check the Prolific dashboard: a *draft* may already exist, so delete it before retrying |

## Collection

| message / symptom | cause | what to do |
|---|---|---|
| log stays at `Prolific poll: … completed=k target=N` | recruitment slow | normal for up to 2 h, then it moves on with partial data **and leaves the study open** |
| ``live collection needs `prolific_study_id` in the experiment config …`` | `--mode live` without a Prolific study (e.g. `--prolific-mode none` on a fresh experiment) | live collection needs a study |
| `mode='live' requires a deployed experiment to collect from …` | no `results_api_url` in `experiment/config.json` | the deploy did not complete |
| `AUTO_PSYCH_RESULTS_TOKEN is not set — cannot fetch the token-guarded /results endpoint.` | token missing in the collecting environment | add it; recover with `RESUME_AGENTS=4_collect:5_model_loop` |
| `live results fetch failed for …: HTTPError … refusing to report zero responses` | wrong token (403) or network | data are safe in Firestore; fix, then recover as above |
| `Participant collection returned no data (0 rows).` | nothing reached `/submit` | open the page yourself; check the browser console; check the monitor |
| `Collected data failed the quality check: all N responses are identical …` | everyone pressed the same side: broken button mapping, or a bot farm | inspect the page and the monitor. The data were **not** written for modelling. |
| `CERTIFICATE_VERIFY_FAILED` | Python cannot find the system certificates | `_env.sh` sets `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`; make sure it was sourced |

## Model stage (`5_model_loop`)

| message / symptom | cause | what to do |
|---|---|---|
| `ValueError: …/model_loop/responses.csv has column(s) ['participant_id_str', 'chose_right', 'model'] beyond the raw ones …` | code from before 28 September 2026 (runbook § 0 A, fixed) | copy the current code into the run copy; pooling now keeps only the five raw columns. Recover with `RESUME_AGENTS=5_model_loop` (§ 10) |
| `response row … lacks raw column(s) [...]` | a collected row, or a `data/responses.csv`, is missing one of the five raw columns | inspect the collected file in `raw_collected/`; the collector or `/results` changed |
| `No response rows found for inner loop under …` | no `data/responses.csv` in any experiment so far | collection did not finish |
| a proposal rejected with a reason (in `attempted_hypotheses.jsonl` and the round's directory) | expected: failed code check, no convergence, too slow (15 min), or near-duplicate | nothing, unless *every* proposal is rejected every round |
| `history.json` shows `"no_critique"` for a round | critique agent wrote no usable statistic twice, or none produced a p-value | the round still ran; look at `iter_<i>/critique/` |
| `SQLITE_CORRUPT: database disk image is malformed` (opencode) | two runs sharing opencode's database | `run_live.sbatch` gives each run a private one; don't run agents outside it |
| `Stale file handle` from PyTensor | shared compile cache on NFS | `run_live.sbatch` puts it on node-local disk; same advice |
| job killed for memory (`OUT_OF_MEMORY` in `sacct`) | pooled multi-experiment fits are memory-heavy | raise `--mem` (the job asks for 64 GB) and recover with `RESUME_AGENTS` |
| job hits its time limit | too many rounds or proposals for `walltime` | recover the remaining stages with `RESUME_AGENTS`; raise `walltime` or `qos: long` next time |

## Monitor, viewer and collection of results

| message | cause | what to do |
|---|---|---|
| `Firestore read needs Application Default Credentials. Run: gcloud auth application-default login && …` | no Google credentials on this computer | run the two commands it prints (on your own computer) |
| monitor shows no studies | `--data-root` has no `deployment/deployment_manifest.json` with `deploy_target: firebase` | rsync the manifests again (runbook § 7); dry-run deploys are never shown |
| Prolific error shown inline in the monitor | no `PROLIFIC_API_TOKEN` on that computer, or Prolific down | the participant data still display |
| `ERROR: interpreter '…/.venv/bin/python' not found — set PY=…` | `collect_results.sh` defaults to a checkout `.venv` | `PY=$SCRATCH/auto-psych/outer_loop_live/venv/bin/python` |
| collector refuses a non-empty destination | `DEST` already exists | pick a new `DEST` (the default holds earlier results); `--overwrite` replaces them |
| collector raises because a Prolific ID survived | an ID in an unexpected place | do not work around it: find the file and extend the scrubber |

## When something on the cluster itself looks broken

If the scheduler, a filesystem, modules or the network look broken, send the
symptom, hostname, time and job id to srcc-support@stanford.edu. Do not try to
work around it. Prolific studies still need stopping by hand.

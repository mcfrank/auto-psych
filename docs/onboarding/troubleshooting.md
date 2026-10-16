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
| `… has no code_provenance.json; cannot record deployment provenance` | a hand-made run copy, or code rsynced into one | record the commit (runbook § 9); nothing was created |
| ``live collection needs `prolific_study_id` …`` | a fresh experiment resumed with `PROLIFIC_MODE=none` | live collection needs a study |
| `Claude agents need a stated billing mode …` / `… is not set` | no `claude_auth`, or no key for it | runbook § 1 |
| `[USAGE LIMIT] … Waiting N min` | the agents' account hit its usage or rate limit | normal: the agent reruns after the reset |
| `AgentUsageLimitExceeded` / `AgentLoginFailed` | a limit outlasting 12 h; a login that cannot pay | fix the account; `RESUME_AGENTS=5_model_loop` |
| `… is a bare list of names: this run started on code that protected its starting models …` | continuing a run started before 28 September 2026 | finish it on its own code, or start a new run |
| `FATAL: bwrap not on PATH …` | the `system bubblewrap` module did not load | `ml spider bubblewrap`; if the module is broken, that is a cluster problem (below) |
| `Error: --prolific-mode live … Pass --confirm-live-recruitment` | `run.py` or `run_live.sbatch` called by hand | `CONFIRM_LIVE_RECRUITMENT=1`, or `PROLIFIC_MODE=none` for a recovery |
| `No Prolific study settings at …` / `… sets total_available_places: …, but this run was started with --n-participants …` | study settings not rendered, or a different count | render with `_pilot_config.py <config> --render-only` in the checkout the job runs from; keep one count |
| `The researchers' raw collected data would be written to …` | `run.py` by hand with the output tree inside the repository | set `AUTO_PSYCH_OUTPUT_DIR` outside it |
| `LiveStudyAlreadyRecorded: experiment<N> already has a live Prolific study …` | relaunch of an experiment with a live study | nothing ran. `RESUME_AGENTS=4_collect:5_model_loop`; a second study only after stopping the first, with `PUBLISH_ANOTHER_PROLIFIC_STUDY=1` |
| `[error] Experiment N's model loop is not complete (…)` | the previous experiment's model stage did not finish | rerun it (`RESUME_AGENTS=5_model_loop` for that experiment) |
| `[error] 3_implement still invalid after 3 attempt(s): …` | the agent could not make a valid page | read `experiment<N>/logs/3_implement.jsonl` and `logs/.xdg_data/opencode/log/opencode.log`; nothing was deployed |
| `opencode.log`: `You exceeded your current quota … free_tier_requests, limit: …` | the Gemini key is on the free tier (limit 0 for pro; 5/min, 20/day for flash) | a key from a billing-enabled project (runbook § 1); nothing was deployed |
| `curl: (35) … same issuer/serial …` / `HTTP 000` on a login node | el7's `curl` cannot talk to the site | not the site: check with Python (runbook § 5, R3) or a browser |
| `AUTO_PSYCH_RESULTS_TOKEN is not set …` | missing key | at deploy time nothing was created; at collection time the study is already running: add it, recover with `RESUME_AGENTS=4_collect:5_model_loop` |
| `Firebase deploy failed …` / `… the experiment page is NOT live …` | expired `FIREBASE_TOKEN`, permissions, outage, hosting not published | no study was created: fix the cause and relaunch |
| `The functions deploy exited 0 but firebase-tools could not read functions/index.js …` | the deploy ran without `_env.sh`'s `node` wrapper first on `PATH` | no study was created: run from a shell or job that sourced `_env.sh` and relaunch |
| `The deployed /results answered a read without the token with 400 …` / `… with the token with 403 …` | the functions were not replaced / they hold a different `AUTO_PSYCH_RESULTS_TOKEN` | no study was created: redeploy, or use the token the functions were deployed with |
| `Failed to create Prolific study: …` | often insufficient funds | the page is live, no study exists: fix, relaunch |
| `Could not list the account's Prolific studies to exclude earlier participants; no study was created: …` | Prolific API error while reading the earlier studies | the page is live, no study exists: relaunch once Prolific answers |
| `Failed to create Prolific study: …` naming `previous_studies_blocklist` | an earlier pipeline study is in another Prolific project (the filter takes only studies of the new study's project) | the page is live, no study exists: move that study into the account's default project in the dashboard, relaunch |
| `Prolific filter 'previous_studies_blocklist' … is missing from GET /filters/ …` | Prolific renamed or changed the filter that excludes earlier participants | nothing was deployed: find it in `GET /filters/` and update `EARLIER_STUDIES_BLOCKLIST_FILTER` (`deployment/prolific.py`) |
| `Failed to publish …` | often insufficient funds | a draft is recorded: check it in Prolific; recover with `RESUME_AGENTS` or `PUBLISH_ANOTHER_PROLIFIC_STUDY=1` after deleting it |
| study `AWAITING REVIEW` with `AUTOMATICALLY_APPROVE`; some participants unpaid | Prolific holds back submissions faster than its threshold (`time_taken_under_auto_approval_threshold`; under about 3.5 of 7 minutes) | approve or reject them in the dashboard (runbook § 10); the pipeline never does |
| log stays at `Prolific poll: … completed=k target=N` | slow recruitment | normal for up to 3 h, then the study is paused |
| `PAUSED Prolific study <id>: collection gave up at k/N …` | 3 hours passed | expected; the partial data are modelled |
| `Could not pause Prolific study …` (or it `… is in state '…'`) | API error or unusual state | **pause it in the dashboard now**, then recover |
| `live results fetch failed …` | wrong token or network | data are safe in Firestore; fix, recover with `RESUME_AGENTS=4_collect:5_model_loop` |
| `Collected data failed the quality check: all N responses are identical …` | broken buttons or bots | inspect the page and the dashboard; nothing was modelled |
| a carried model `dropped` or a proposal `rejected` with `non-finite ELPD-LOO (nan)` | before 2026-10-07: arviz's PSIS turned a trial whose log-likelihood is constant only to rounding (`p_left` = 0.5 ± an ulp on a mirror pair) into NaN; the model was fine | fixed in `loo_reliability.loo_diagnostics`; on current code a NaN means a real non-finite log-likelihood (a `p_left` of exactly 0 or 1 against an observed answer) |
| proposal rejected with a reason (e.g. `too slow to fit … 30-minute limit`) | a failed admission check | expected, unless every proposal fails every round |
| `filelock … Timeout … .lock` (PyTensor) | processes sharing a compile directory | each job and fit has its own; do not set `compiledir=` in `PYTENSOR_FLAGS` |
| `"no_critique"` in `history.json` | the critic produced no usable statistic | the round still ran; see `iter_<i>/critique/` |
| `OUT_OF_MEMORY` in `sacct`, or the time limit | large pooled fits, many rounds (the model loop holds every loaded fit: about 1.5 GB a model at 7,680 trials) | raise `--mem` (the live job asks for 128 GB; 64 GB ran out in experiment 3) or `walltime`; recover with `RESUME_AGENTS` (the model loop restarts from its recorded input) |
| a model-loop round takes hours and the job log shows the same model's `Sampling 4 chains …` again and again | a run copy older than e6c6759 (2026-09-30): its model stage had no fit cache, so every novelty check re-sampled every admitted model | move the run onto current code ([runbook § 9](running_a_live_experiment.md#9-recovering-without-paying-again)) and resume with `RESUME_AGENTS=5_model_loop`; `experiment<N>/model_loop/.fit_cache/` then gains one `.nc` per fit |
| `sbatch: error: … timelimit request too short for QOS long` or `Invalid qos specification` | `--qos=long`, which this account does not have | nothing was submitted: leave `--qos` out; the `mcfrank` partition allows up to 7 days (`normal` caps at 48 hours) |
| design: `No models in … can be evaluated on the design pool's stimulus rows (… e.g. participant_id …)` | code from before 2026-09-30, whose design after data dropped models with a participant effect (on human data, often every one) | nothing was deployed: move the run onto current code ([runbook § 9](running_a_live_experiment.md#9-recovering-without-paying-again)); move the half-made `experiment<N>/` aside first if it has no study and no data |
| the model stage ends with `[inner-loop] Carried the live set … set = ['<one model>']` | every rival was more than `prune_dse_multiplier`·clustered SE behind; a design over one model has nothing to discriminate (its EIG is zero for every pair) | before the next experiment starts: `python -m src.pipelines.outer_loop.reprune --experiment-dir … --dse-multiplier <m>` with the run's sampler settings (a job, not a login node), and the same multiplier in the config for what follows |

**If the cluster itself looks broken** (scheduler, filesystem, modules,
network): stop, note the symptom, hostname, time and job id, and send them to
srcc-support@stanford.edu yourself. Prolific studies still need pausing by
hand.

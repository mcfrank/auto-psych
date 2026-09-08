# Recovery-improvement campaign (Sherlock)

A batch job in which a **Claude Code agent reviews the most recent
holdout-recovery sweeps, prescribes improvements to the model-recovery loop,
implements one of them on its own branch, and launches a sweep to test it**.
When the sweep finishes, a fresh review job reads its results and iterates.
A campaign runs unattended for days; nothing in it holds an allocation idle.

```
start_campaign.sh ──► review job (iter1) ──► sweep 1 (setup → 20-task array → analysis)
                                                │ afterany
                          review job (iter2) ◄──┘ ──► sweep 2 ──► review job (iter3) ──► sweep 3 ──► finalize job
```

Each review job runs `claude -p` headlessly on a compute node (your
subscription login over NFS, no API key), on a git clone of the repo branched
per iteration. Its deliverables are on-disk artefacts the wrapper validates
before it spends anything: a `prescription.md`, commits on the branch, and a
`next_run.env` (which sweep to launch) or `STOP`.

## Start

```bash
# commit your code first — the agent clones HEAD of the current branch
bash scripts/recovery_improvement/start_campaign.sh sept_2026
```

That reviews the four most recent faithful sweeps on `$SCRATCH`
(`holdout_faithful_32eig_32random` is the reference), runs 3 review sessions
with a full 5×4 sweep after each, and writes everything under
`$SCRATCH/auto-psych/recovery_improvement/sept_2026/`. Knobs (env vars) are
listed at the top of `start_campaign.sh`; e.g.

```bash
MAX_ITERATIONS=5 REVIEW_MAX_TURNS=600 \
  bash scripts/recovery_improvement/start_campaign.sh sept_2026_long \
  $SCRATCH/auto-psych/holdout_faithful_32eig_32random $SCRATCH/auto-psych/holdout_faithful_64random
```

`AFTER_JOB=<jobid>` delays the first review until a running sweep has finished.

## Watch / read / stop

```bash
bash scripts/recovery_improvement/campaign_status.sh sept_2026   # iterations, jobs, per-GT means
squeue --me
tail -f $SCRATCH/auto-psych/recovery_improvement/sept_2026/slurm_logs/review_iter1_<jobid>.out
cat  $SCRATCH/auto-psych/recovery_improvement/sept_2026/journal.md
cat  $SCRATCH/auto-psych/recovery_improvement/sept_2026/iter1/prescription.md
bash scripts/recovery_improvement/stop_campaign.sh sept_2026 [--cancel-sweeps]
```

## Layout of a campaign root

```
campaign.env          pinned settings (written once by start_campaign.sh)
journal.md            one entry per iteration: what was reviewed, decided, launched
iter<N>/
  digest.md           the condensed sweep results the agent was shown
  prompt.md           the exact brief it received (prompt.repair1.md if repaired)
  claude_stream.jsonl the raw session stream; token_usage.jsonl its spend
  repo/               git clone on branch recovery-improvement/<name>/iter<N>
  prescription.md     the agent's findings / diagnosis / ranked prescriptions / change
  next_run.env | STOP its decision
  jobs.json           sweep job ids + the chained review job
  sweep/              WORK_ROOT of the sweep that tested this iteration's change
                      (test_retest.json, run<r>/<gt>/holdout.*, slurm_logs/ ...)
final_digest.md       written by the finalize job after the last sweep
slurm_logs/           the review jobs' logs
```

## Getting the improvements back

Each iteration's branch lives only in its clone. To bring one into your checkout:

```bash
git fetch $SCRATCH/auto-psych/recovery_improvement/sept_2026/iter3/repo recovery-improvement/sept_2026/iter3
git log --oneline main..FETCH_HEAD          # review the agent's commits
git checkout -b improvements FETCH_HEAD     # or cherry-pick
```

Iteration N's branch contains iterations 1..N-1's commits too.

## What the agent may and may not do

The brief (`review_prompt.md`) tells it: improve the loop generally (candidate
prompts, admission/novelty/pruning, critique, design, sampler settings,
robustness); never touch the ground-truth registry, the model-family twins, the
evaluation code, or the held-out parameters; one coherent change per iteration;
keep the sweep comparable (same GTs, 5 repeats, `BASE_SEED=100`); fail loudly;
commit everything; never push, never cancel jobs, never wait on jobs. Claude
Code additionally denies `scancel`, `scontrol update/hold/release/requeue` and
`git push`. Each session is capped by `REVIEW_MAX_TURNS`, `REVIEW_MAX_BUDGET_USD`
(an *estimated* cost cap — under a subscription login it limits session size, it
is not billed) and `REVIEW_TIMEOUT_SEC`.

Sweeps are launched through the iteration repo's own
`scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh`, so the
agent's edits to the sweep scripts and configs take effect. The sweep's own
agents still use opencode + Gemini from `.secrets` (`GOOGLE_API_KEY`), as before.

## Failure handling

- Invalid deliverables → one repair session with the problems injected → then
  the review job fails and the chain stops (nothing is launched). Fix or delete
  `iter<N>/next_run.env`/`STOP` and resubmit that iteration:
  `sbatch --export=ALL,CAMPAIGN_ROOT=<root>,ITERATION=<N>,MODE=review scripts/recovery_improvement/review_iteration.sbatch`
  (add the same `--partition/--time/--output` flags `start_campaign.sh` uses).
- A sweep whose setup fails is killed rather than left pending
  (`SBATCH_KILL_INVALID_DEP=yes`); the next review still runs (`afterany`) and
  sees the failure in its digest, which is exactly what it should fix.
- The review job's Slurm walltime (`REVIEW_TIME`, 8 h) must exceed
  `REVIEW_TIMEOUT_SEC` (6 h) so validation and launching still happen after a
  timed-out session.

## Dry run

To see the digest and brief iteration N would get, without spawning Claude or
submitting anything (on a compute node):

```bash
srun -p dev -c 1 --mem 4G -t 00:10:00 bash -c '
  ml load system uv; export VENV_PY=<campaign root>/venv/bin/python
  $VENV_PY scripts/recovery_improvement/run_iteration.py --campaign-root <campaign root> --iteration 1 --dry-run'
```

Output lands in `<campaign root>/dryrun_iter<N>/`.

## Code

- `src/recovery_improvement/` — `campaign.py` (campaign.env), `digest.py` (sweep →
  Markdown), `next_run.py` (the `next_run.env` contract), `iteration.py` (the
  iteration flow, with the agent and Slurm injected), `slurm.py` (the real
  Claude / sbatch adapters).
- `scripts/recovery_improvement/run_iteration.py` — the CLI the sbatch calls.
- Tests: `tests/test_recovery_improvement.py`, `tests/test_recovery_improvement_units.py`.

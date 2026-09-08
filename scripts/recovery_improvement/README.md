# Recovery-improvement review (Sherlock)

A Slurm job in which a **Claude Code agent reviews the most recent
holdout-recovery sweeps, writes a prescription for improving the model-recovery
loop, implements one change on its own branch, and prepares the recovery sweep
that would test it**. It launches nothing. You read the prescription and decide
whether to launch that sweep; when the sweep has finished, you run the review
again and the agent iterates on the new results.

```
review.sh <name>  ──►  review job (iter1): digest → Claude → prescription.md + branch + next_run.env
                                                   │
        you read iter1/prescription.md;  launch_next.sh <name>  ──►  sweep (setup → 20-task array → analysis)
                                                   │
review.sh <name>  ──►  review job (iter2): digests iter1's sweep vs. the baseline → …
```

The review job runs `claude -p` headlessly on a compute node using your
subscription login over NFS (no API key; the sbatch strips any it might inherit).
It works in a git clone branched per iteration, so your checkout is never touched.

## Use

```bash
# commit your code first — the agent clones HEAD of the current branch
bash scripts/recovery_improvement/review.sh sept_2026        # submits the review job (~hours)

bash scripts/recovery_improvement/campaign_status.sh sept_2026
cat  $SCRATCH/auto-psych/recovery_improvement/sept_2026/iter1/prescription.md
cat  $SCRATCH/auto-psych/recovery_improvement/sept_2026/journal.md

bash scripts/recovery_improvement/launch_next.sh sept_2026   # only if you want the sweep it proposes
bash scripts/recovery_improvement/review.sh sept_2026        # once that sweep has finished: iteration 2
bash scripts/recovery_improvement/stop_campaign.sh sept_2026 # cancel a queued/running review (+ --cancel-sweeps)
```

The first call reviews the four most recent faithful sweeps on `$SCRATCH`
(`holdout_faithful_32eig_32random` is the reference, since it matches the
current config's 32+32 design); pass other sweep roots as extra arguments.
Knobs (model, turn/time caps, sweep defaults) are listed at the top of
`review.sh` and pinned into `campaign.env` on the first call.

## What you get per iteration

```
$SCRATCH/auto-psych/recovery_improvement/<name>/
  campaign.env         pinned settings
  journal.md           one entry per iteration: reviewed / decided / launched
  iter<N>/
    digest.md          the condensed sweep results the agent was shown
    prompt.md          the exact brief it received
    prescription.md    findings, diagnosis, ranked prescriptions, the change made,
                       risks, leakage statement
    repo/              git clone on branch recovery-improvement/<name>/iter<N>
                       (iteration N's branch contains iterations 1..N-1's commits)
    next_run.env       the sweep it proposes (KEY=value overrides of the defaults) — or STOP
    jobs.json          the resolved sweep env + the launch command
    claude_stream.jsonl / token_usage.jsonl   the raw session and its spend
    sweep/             appears only after YOU run launch_next.sh (test_retest.json, run<r>/<gt>/holdout.*, slurm_logs/)
  slurm_logs/          the review jobs' logs
```

To keep an iteration's changes:

```bash
git fetch $SCRATCH/auto-psych/recovery_improvement/sept_2026/iter2/repo recovery-improvement/sept_2026/iter2
git log --oneline main..FETCH_HEAD
git checkout -b improvements FETCH_HEAD     # or cherry-pick
```

## What the agent may and may not do

`review_prompt.md` is its brief: improve the loop generally (candidate prompts,
admission/novelty/pruning, critique, design, sampler settings, robustness);
never touch the ground-truth registry, the model-family twins, the evaluation
code, or the held-out parameters; one coherent change per iteration; keep the
proposed sweep comparable (same GTs, 5 repeats, `BASE_SEED=100`); fail loudly;
commit everything; never launch sweeps, push, or cancel jobs. Claude Code
additionally denies `scancel`, `scontrol update/hold/release/requeue` and
`git push`. Each session is capped by `REVIEW_MAX_TURNS`, `REVIEW_TIMEOUT_SEC`
(6 h inside an 8 h job) and `REVIEW_MAX_BUDGET_USD` (an *estimated* cost; under a
subscription login it only limits session size).

The wrapper validates the deliverables before the job ends: `prescription.md`,
a clean committed tree, and exactly one of `next_run.env` (checked against the
allowed keys and the config's ground truths) / `STOP`. Invalid ones get one
repair session with the problems injected; then the job fails and
`campaign_status.sh` shows why.

## Launching the sweep

`launch_next.sh` merges `next_run.env` over the campaign's sweep defaults and
runs the iteration repo's own `scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh`,
so the agent's edits to the Slurm scripts and configs take effect. It asks for
confirmation (`YES=1` skips), records the job ids in the journal, and writes
the sweep to `iter<N>/sweep/`. A full sweep is 20 tasks of 3–5 h, five at a
time (about a day), and spends roughly $165 of Gemini through the sweep's own
opencode agents (`GOOGLE_API_KEY` in `.secrets`, as before).

## Unattended mode (opt-in)

`AUTO_LAUNCH=1 bash scripts/recovery_improvement/review.sh <name>` makes every
review job launch its proposed sweep itself and chain the next review
`afterany` the sweep's analysis job, up to `MAX_ITERATIONS`, ending with a
`finalize` job that writes `final_digest.md`. Nothing watches that chain: if a
review job dies, the chain stops and `campaign_status.sh` is how you notice.

## Dry run

To see the digest and brief a review would get, without spawning Claude or
submitting anything (on a compute node):

```bash
DRY_RUN=1 bash scripts/recovery_improvement/review.sh trial      # writes campaign.env only
srun -p dev -c 1 --mem 4G -t 00:10:00 bash -c '
  export VENV_PY=$SCRATCH/auto-psych/dev_test_env/venv/bin/python
  $VENV_PY scripts/recovery_improvement/run_iteration.py \
    --campaign-root $SCRATCH/auto-psych/recovery_improvement/trial --iteration 1 --dry-run'
```

## Code

- `src/recovery_improvement/` — `campaign.py` (campaign.env), `digest.py` (sweeps →
  Markdown), `next_run.py` (the `next_run.env` contract), `iteration.py` (the
  review flow), `slurm.py` (the Claude / sbatch adapters).
- `scripts/recovery_improvement/run_iteration.py` — the CLI the sbatch calls.
- Tests: `tests/test_recovery_improvement.py`, `tests/test_recovery_improvement_units.py`.

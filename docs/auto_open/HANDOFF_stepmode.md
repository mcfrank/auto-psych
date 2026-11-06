# Handoff: step mode on Sherlock (2026-11-04)

For the Sherlock session. Step mode is built on `auto-psych-open` (`DISTRIBUTED_DESIGN.md` §0b,
"Step mode as built"). Nothing here recruits anyone. Run on the lab's allocation; report back in
`DISTRIBUTED_DESIGN_REVIEW_V3.md` §11 (or a new review file).

## 1. Slow tests (a job on `mcfrank` or `normal`, 16 cores, 64 GB, 6 h)

```bash
uv run pytest -q tests/test_rsa_open_step_equivalence.py tests/test_rsa_open_loop_orchestrator.py \
  tests/test_rsa_open_fit_array.py tests/test_rsa_open_loop_scores.py
```
The first is the step-vs-straight equivalence test (two two-round loops, every fit twice). Report
pass/fail, wall time, and, on a failure, the first differing ledger line or history step.

## 2. A small step-mode rehearsal (simulated participants)

One chain, small, in a fresh work root, so nothing touches the live campaign's:
```bash
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_steps_rehearsal
STEP_MODE=1 CHAIN=0 ARM=open N_EXPERIMENTS=2 MAX_ITERATIONS=2 CANDIDATES=3 \
  sbatch --job-name=rsa_open_reh_c0 --partition=mcfrank --cpus-per-task=8 --mem=32G --time=2-00:00:00 \
    --no-requeue -o $WORK_ROOT/logs/%x_%j.out scripts/rsa/slurm/open_rehearsal.sbatch
```
(`mkdir -p $WORK_ROOT/logs` first.) It should run to its end without anyone touching it. Check and
report:
- `cells/open_chain_0/steps.jsonl`: steps per round (expect 3-5), arrays per step, task counts;
  the time each step waited for its arrays (queue + run) and ran itself.
- No STOP, no step that died (`sacct --name=rsa_open_reh_c0`), and every array task's exit 0
  except infrastructure (`sacct --name=rsa_open_reh_c0_fits`); a theory's failure is a result
  file, not a failed task.
- `step_settings.sh` and `continuation.sh` in the cell look right; every step ran on `mcfrank` and
  every fit on `mcfrank` or `owners` with `CPU_GEN:SIE`.
- Each round's `journal.json`, and that no agent directory was written twice (one session per
  `candidate_*`, `critique`), and the ledger has no duplicate lines.
- Fairshare: `sshare -U -l` before and after.
- Peak memory of the steps (`sacct -o MaxRSS`): is 32 GB right? Fits: is 18 GB enough?

Then, if it went well, the same with the defaults (`N_EXPERIMENTS=4 MAX_ITERATIONS=5 CANDIDATES=6`)
in parallel with a second chain (`CHAIN=1`, another `--job-name`). Kill one step mid-round once
(`scancel` while theorists run) and relaunch it with the same command: it should continue the
round from its journal without rerunning finished agents.

## 3. Still open from §10
- A5 (compile separately, then sample at ~5.5 GB): a prototype if time allows.
- Whether `owners` time counts toward fairshare (the `sshare` readings above).

## 4. Next: one experiment of a real-agent chain (added 2026-11-05)

SRCC's latest reply (ticket #106983, 11-05) is in `DISTRIBUTED_DESIGN.md` §0b ("After SRCC's
reply"): not a ruling, "try to run some segments", report any rejection's exact message, and avoid
`sbatch --wait`. Code: `auto-psych-open` after this handoff's commit (one experiment per launch, 40
steps a launch, folds after their full fits, 18/36/72 GB tiers, the §11 findings fixed).

**The PI submits the launch by hand**; this session doesn't submit agent-running jobs (rule 10).
It reads results afterwards, with no polling loops.

1. Fast tests: `uv run pytest -q tests/test_rsa_open_step_*.py tests/test_rsa_open_fit_array.py`.
   Slow: `tests/test_rsa_open_step_equivalence.py` (its limit is now set; expect it to pass as your
   corrected copy did, in fewer steps since the seeds' folds follow their fits).
2. **The smoke-size replay only** (§11's smoke: 1 starting theory plus the chance model, 1 round of
   2; about 6.5 h, mostly queue waits), as a mechanics check before agents are spent: a fold array
   held until its full fits end, then released and run; `launch.json` written by the first step and
   read by the rest; DONE at the end (the replay is one experiment, so it ends the chain rather than
   stopping for the next launch). Not the full or second-chain replays, and no step-timing analysis:
   without agents every step is short, so their durations say little about live ones (PI, 11-05).
   Report only whether those mechanics worked, with the job ids of anything that didn't.
3. The real-agent segment, launched by the PI: one open chain, simulated participants, one
   experiment of two (`STEP_MODE=1 CHAIN=0 ARM=open N_EXPERIMENTS=2 MAX_ITERATIONS=2 CANDIDATES=3` with
   `open_rehearsal.sbatch`, as in §2). Report: steps and their durations, step MaxRSS with agents,
   whether any session ran twice, any message from Slurm, the agent or SRCC, and that it stopped
   after the experiment with a mail.

   The PI's command, from a Sherlock login shell in the checkout at `auto-psych-open` (a fresh work
   root, so the staged code is this commit's; copying `private/ground_truth/` from an earlier
   rehearsal root first saves rebuilding it):
   ```bash
   export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_stepped_1 && mkdir -p $WORK_ROOT/logs
   STEP_MODE=1 CHAIN=0 ARM=open N_EXPERIMENTS=2 MAX_ITERATIONS=2 CANDIDATES=3 \
     sbatch --job-name=rsa_open_stepped_c0 --partition=mcfrank --cpus-per-task=8 --mem=32G \
       --time=2-00:00:00 --no-requeue -o $WORK_ROOT/logs/%x_%j.out scripts/rsa/slurm/open_rehearsal.sbatch
   ```
   Later steps go to `STEP_PARTITION` (`mcfrank` unless set). If chain 0's live job holds the lab
   node, `STEP_PARTITION=normal` (exported before `sbatch`) rather than `owners`: a step is never
   requeued, so a preempted step would end the segment mid-session.

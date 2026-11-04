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

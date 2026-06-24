# OOM watchdog

Watches **all your other jobs** and, when one is killed for running out of
memory, **resubmits that exact task with more memory**. An instance of Claude
Code makes the judgement calls (is this really an OOM? how much memory? give up
yet?); a shell toolkit (`wd.sh`) does the deterministic Slurm mechanics.

Because Sherlock forbids long-lived "sleeper" jobs (one that holds an allocation
while idle between checks), the watchdog runs as a **short tick every few minutes
via `scrontab`** (Slurm cron) instead of one looping job. Each tick scans and
acts, then exits — nothing sits idle holding a node.

## Quick start

```bash
bash scripts/watchdog/install_watchdog.sh
```

This registers a scrontab entry. The first tick snapshots every job you own
(running + pending) and watches them from then on.

```bash
scrontab -l                                   # confirm it's scheduled
tail -f <state_dir>/tick.out                  # follow it (path is printed on install)
<toolkit>/wd.sh status                        # what it has done
bash scripts/watchdog/uninstall_watchdog.sh   # stop it
```

The watchdog **retires itself** once none of the watched jobs are left
queued/running (it then no-ops cheaply until you uninstall it).

## How it resubmits OOM'd jobs *correctly*

The hard part of "resume with more memory" is reproducing the failed task
*exactly* — same seed, same ground-truth model, same flags — otherwise your
pipeline's `--resume` would write to the wrong cell. The watchdog captures each
running job's **full submit environment** on the first tick, read straight from
the live batch process:

```
srun --jobid=<job> --overlap --input=none ... base64 /proc/<batch-pid>/environ
```

That recovers the real `WORK_ROOT`, `BASE_SEED`, `INNER_LOOP_ITERATIONS`,
`GT_MODELS`, etc. — not script defaults (your jobs ran with `BASE_SEED=100`, so
defaults would be wrong). On OOM it resubmits the same `--array=<task>` against
the same script with that environment replayed, so `--resume` continues the task
instead of restarting it.

> A task can only be auto-resubmitted if its recipe was captured while it was
> running. If a task OOMs before its recipe was captured, the watchdog **refuses
> to guess** and logs `no-recipe`. So install it while your jobs are alive.

## The escalation policy

| OOM # | memory      | partition | notes                         |
|-------|-------------|-----------|-------------------------------|
| 1     | 32 → 64 GB  | `normal`  | `normal` caps at 8 GB/core    |
| 2     | 64 → 128 GB | `bigmem`  |                               |
| 3     | 128 → 256 GB| `bigmem`  |                               |
| 4     | —           | —         | give up, log `exhausted`      |

Memory **doubles** each time; above 64 GB it moves to `bigmem`; the hard cap is
256 GB; max 3 retries. If a task's measured `MaxRSS` already exceeds the doubled
value, Claude jumps to the next size that comfortably clears it. Only genuine
memory failures are acted on — a `FAILED`/exit-125 that turns out to be a code
bug is confirmed against the job log and skipped, not resubmitted.

## Knobs (environment variables, set when you run `install_watchdog.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `WD_STATE_DIR` | `$SCRATCH/auto-psych/watchdog/<timestamp>` | recipes, ledger, logs |
| `WD_PARTITION` | `normal` | partition for the tick job (`hns`, `dev`, …) |
| `WD_POLL_MINUTES` | `5` | minutes between ticks |
| `WD_TICK_WALLTIME` | `00:20:00` | max time for a single tick |
| `WD_MODEL` | `claude-sonnet-4-6` | model the recovery operator uses |
| `WD_MEM_CAP_GB` | `256` | hard memory ceiling |
| `WD_MAX_RETRIES` | `3` | resubmits per task before giving up |
| `WD_BIGMEM_THRESHOLD_GB` | `64` | escalate to `bigmem` above this |

The tick job is tiny (1 CPU / 4 GB) and short, so it backfills immediately — it
never waits behind your big jobs, and you have no per-user job cap. `WD_PARTITION=hns`
runs it in your contributed H&S pool if you'd rather keep it off the `normal`
queue entirely.

## Files

- `install_watchdog.sh` — register the scrontab schedule (run this).
- `uninstall_watchdog.sh` — remove the schedule and cancel any running tick.
- `watchdog_tick.sh` — one tick: capture → scan → (Claude only if an OOM appears).
- `wd.sh` — toolkit: `init`, `capture`, `scan`, `resubmit`, `handle`, `status`.
- `watchdog_prompt.md` — the recovery operator's instructions.

## Safety

- Claude runs with a tight tool allowlist: `wd.sh` (the only state-changing
  command) plus read-only inspection. It cannot cancel, hold, or alter jobs.
- `wd.sh resubmit` validates every request (memory cap, per-core feasibility on
  the chosen partition) and **fails loudly** rather than submitting something
  invalid.
- Everything it does is appended to `<state_dir>/ledger.jsonl`; nothing is silent.
- Dry-run any resubmit yourself: `WD_DRY_RUN=1 wd.sh resubmit ...` prints the
  exact `sbatch` it would run without submitting.
- Auth uses your `~/.claude` credentials over NFS; compute nodes here have
  direct outbound internet, so no API key is needed.

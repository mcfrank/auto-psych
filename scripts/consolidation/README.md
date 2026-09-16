# Consolidation job (Sherlock)

Runs `docs/consolidation_plan_2026_09.md` unattended: a Claude Code agent
(Opus 4.6 by default, `MODEL=` overrides) executes the plan **one phase per session** in its own clone
on a compute node. Nine phases: baseline, integrate (merge `main`, restore the
leakage audit, merge arm C), true-raw inner loop + verifier, honest metrics +
offline diagnostics, race presentation, lens rotation, docs + full checks,
two SMOKE cells, verdict + handoff. It launches nothing but those two smoke
cells; the 24-cell comparison sweep is written into `HANDOFF.md` for you.

```bash
bash scripts/consolidation/consolidate.sh              # submit (commit first)
cat  $SCRATCH/auto-psych/consolidation_2026_09/STATUS.md
ls   $SCRATCH/auto-psych/consolidation_2026_09/progress/   # P<k>.done / P<k>.blocked
cat  $SCRATCH/auto-psych/consolidation_2026_09/HANDOFF.md  # when P8 is done
```

Layout of the work root:

```
consolidation.env      frozen inputs (SHAs), model, caps, Slurm settings
repo/                  the clone, branch consolidate/2026-09 (refs/consolidation/* = inputs)
progress/              P<k>.done | P<k>.blocked, baseline_failing_tests.txt,
                       P<k>.session<n>.{prompt.md,jsonl}, smoke_jobs.json, token_usage.jsonl
smoke_featurized/ smoke_raw/   the P7 smoke cells (+ VERDICT.md from the verifier)
STATUS.md              one line per event (session start/end, requeue, phase done)
VERDICT.md HANDOFF.md  written by P8
slurm_logs/            the job's logs (one per requeue)
```

The driver (`run_consolidation.py`, pure parts in `src/consolidation/driver.py`)
validates each phase marker (HEAD matches, tree clean, phase files present),
gives one repair session, and otherwise stops with `P<k>.blocked`. It never
sleeps: it requeues itself after a session limit (`--begin` at the reset),
when the walltime cannot fit another session, and `afterany` the smoke jobs.
A stopped job is resumed with `RESUME=1 bash scripts/consolidation/consolidate.sh`
after you remove or resolve the `.blocked` marker.

Claude Code denies `scancel`, `scontrol …`, `git push` in every phase and
`sbatch` in every phase but P7.

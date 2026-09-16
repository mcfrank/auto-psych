# Consolidation job (Sherlock)

Runs `docs/consolidation_plan_2026_09.md` unattended: a Claude Code agent
(Opus 4.6 by default, `MODEL=` overrides) executes the plan **one phase per session** in its own clone
on a compute node. Twelve phases: baseline, integrate (merge `main`, restore the
leakage audit, merge arm C), true-raw inner loop + verifier, honest metrics +
offline diagnostics, race presentation, lens rotation, docs + full checks,
two SMOKE cells, verdict + handoff, then the 5-repeat recovery sweep of the
arm(s) in `SWEEP_ARMS` (default `raw`), the RMSE evaluation job, and
`RESULTS.md`.

```bash
bash scripts/consolidation/consolidate.sh              # submit (commit first)
cat  $SCRATCH/auto-psych/consolidation_2026_09/STATUS.md
ls   $SCRATCH/auto-psych/consolidation_2026_09/progress/   # P<k>.done / P<k>.blocked
cat  $SCRATCH/auto-psych/consolidation_2026_09/HANDOFF.md  # when P8 is done (smoke verdict)
cat  $SCRATCH/auto-psych/consolidation_2026_09/RESULTS.md  # when P11 is done (RMSE evaluation)
```

Layout of the work root:

```
consolidation.env      frozen inputs (SHAs), model, caps, Slurm settings
repo/                  the clone, branch consolidate/2026-09 (refs/consolidation/* = inputs)
progress/              P<k>.done | P<k>.blocked, baseline_failing_tests.txt,
                       P<k>.session<n>.{prompt.md,jsonl}, smoke_jobs.json,
                       sweep_jobs.json, analysis_jobs.json, token_usage.jsonl
smoke_featurized/ smoke_raw/   the P7 smoke cells (+ VERDICT.md from the verifier)
sweep_raw/             the P9 recovery sweep (run<r>/<gt>/holdout.*, test_retest.*)
analysis/              the P10 evaluation outputs (paired comparisons, oracle, tables)
STATUS.md              one line per event (session start/end, requeue, phase done)
VERDICT.md HANDOFF.md  written by P8;  RESULTS.md  written by P11
slurm_logs/            the job's logs (one per requeue)
```

The driver (`run_consolidation.py`, pure parts in `src/consolidation/driver.py`)
validates each phase marker (HEAD matches, tree clean, phase files present),
gives one repair session, and otherwise stops with `P<k>.blocked`. It never
sleeps: it requeues itself after a session limit (`--begin` at the reset),
when the walltime cannot fit another session, and `afterany` the jobs a phase
submitted (smoke cells, the sweep, the evaluation).
A stopped job is resumed with `RESUME=1 bash scripts/consolidation/consolidate.sh`
after you remove or resolve the `.blocked` marker.

Claude Code denies `scancel`, `scontrol …`, `git push` in every phase and
`sbatch` in every phase but P7, P9 and P10.

The driver code and the plan are read from the user's checkout at each job
start, so committing an extension to `main` reaches a running job at its next
requeue; the phase list may only be extended at the end.

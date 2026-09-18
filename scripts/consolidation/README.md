# Consolidation job (Sherlock)

Runs `docs/consolidation_plan_2026_09.md` unattended: a Claude Code agent
(Opus 4.6 by default, `MODEL=` overrides) executes the plan **one phase per session** in its own clone
on a compute node. Thirty-two phases: baseline, integrate (merge `main`, restore the
leakage audit, merge arm C), true-raw inner loop + verifier, honest metrics +
offline diagnostics, race presentation, lens rotation, docs + full checks,
two SMOKE cells, verdict + handoff; then (amendment of 2026-09-16) raw as
the only mode, an agent tree with no feature code plus an import gate at
admission, a simplification pass for human readability, a second smoke + verdict, the
5-repeat recovery sweep, the RMSE evaluation job, `RESULTS.md`, and
(amendment of 2026-09-17) an aggressive cleanup — characterize + split the
oversized modules, reduce the surface, readability, an equivalence smoke and
`CLEANUP_REPORT.md`. An amendment of 2026-09-18 inserts P17-P22: sweep 1 ran
with a candidate-write failure (292 of 297 rejections were `no candidate.py
written`), so those phases diagnose and fix it, repair the untested oracle CLI,
re-run the 5-repeat sweep and report it; the cleanup moves to P23-P27. A second amendment the same day adds P28-P31:
sweep 2's empty-round check killed 5 of 20 cells and every formal comparison
returned zero paired cells (the analysis tools never extract an archived run
tree), so those phases make the round recoverable, give the tools
archive-backed tests, and re-analyse both sweeps offline from cached fits,
ending in `ANALYSIS_FINAL.md`.

```bash
bash scripts/consolidation/consolidate.sh              # submit (commit first)
cat  $SCRATCH/auto-psych/consolidation_2026_09/STATUS.md
ls   $SCRATCH/auto-psych/consolidation_2026_09/progress/   # P<k>.done / P<k>.blocked
cat  $SCRATCH/auto-psych/consolidation_2026_09/HANDOFF.md  # when P13 is done (smoke verdict)
cat  $SCRATCH/auto-psych/consolidation_2026_09/RESULTS.md  # when P16 is done (RMSE evaluation)
cat  $SCRATCH/auto-psych/consolidation_2026_09/RESULTS_RERUN.md   # when P22 is done (re-run)
cat  $SCRATCH/auto-psych/consolidation_2026_09/CLEANUP_REPORT.md  # when P27 is done (cleanup)
cat  $SCRATCH/auto-psych/consolidation_2026_09/ANALYSIS_FINAL.md   # when P31 is done (re-analysis)
```

Layout of the work root:

```
consolidation.env      frozen inputs (SHAs), model, caps, Slurm settings
repo/                  the clone, branch consolidate/2026-09 (refs/consolidation/* = inputs)
progress/              P<k>.done | P<k>.blocked | P<k>.retry*, baseline_failing_tests.txt,
                       P<k>.session<n>.{prompt.md,jsonl}, smoke_jobs.json,
                       isolation_smoke_jobs.json, sweep_jobs.json, analysis_jobs.json,
                       token_usage.jsonl
smoke_*/ smoke_round<k>/   the P7 and P12 smoke cells (+ VERDICT.md from the verifier)
sweep/                 the P14 recovery sweep (run<r>/<gt>/holdout.*, test_retest.*)
analysis/              the P15 evaluation outputs (paired comparisons, oracle, tables)
STATUS.md              one line per event (session start/end, requeue, phase done)
VERDICT.md HANDOFF.md  written by P8 and again by P13;  RESULTS.md  written by P16
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
`sbatch` only in the phases that submit jobs (P7, P12, P14, P15, P18, P20, P21, P26, P30). A verdict phase re-opens an
earlier phase by writing `P<j>.retry*` and deleting `P<j>.done` without
writing its own marker; the driver runs `P<j>` again and then the verdict
phase again.

The driver code and the plan are read from the user's checkout at each job
start, so committing an extension to `main` reaches a running job at its next
requeue. Renumbering phases is only safe while no job is running: cancel first,
rename any superseded markers (e.g. `P9_old_launch_sweep.*`), then resubmit.

# Cluster experiments log

This file is appended to by `remote_jobs/submit.py` every time you submit a job.
Each section records what was submitted, where, with which arguments and code commit,
and how to pull the results back. Order is newest-on-top.

Do not edit existing entries by hand; treat this as a journal. New entries land at the
top of the entries list (just below this header) automatically.

<!-- entries:start -->

## 2026-05-08 22:40 UTC -- smoke_e2e
- **Manifest**: `remote_jobs/jobs/smoke_e2e.yaml`
- **Local commit**: `0d56b36` (clean) on `remote_jobs` (pushed, upstream `origin/remote_jobs`)
- **Remote host**: `sherlock` * **Project dir**: `$HOME/auto-psych` * **Scratch dir**: `$SCRATCH/auto-psych`
- **Slurm jobs submitted** (1):
  - `24444568` -- `--project subjective_randomness --mode simulated_participants_nobrowser --n-participants 2 --max-retries 2 --ground-truth-model alternation --runs 1` -- log: `$SCRATCH/auto-psych/logs/smoke_e2e__ground_truth_model-alternation_runs-1-24444568.out`
- **Pull when done**: `./remote_jobs/sync_from_remote.sh --project subjective_randomness`


<!-- entries:end -->

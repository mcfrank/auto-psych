# Handoff: RSA inner loop, Sherlock run 2

For the local session that runs RSA jobs on Sherlock. Run 2 repeats run 1's six
cells (real, recovery_literal, recovery_salience × 2 replicates; 5 rounds × 6
slots) with three changes the PI decided on 2026-10-08 after run 1
(`SHERLOCK_RUN1_RESULTS.md`; overview page `data/rsa/sherlock_run1/overview.html`):

1. **Selection on grouped cross-validation.** The loop now ranks, prunes and
   exports on ELPD-CV: 5 folds of whole training conditions, the unit the test
   set holds out (`src/rsa/loop/cv.py`). In run 1, trial-level PSIS-LOO pruned
   the models that generalised best (real_rep1). Admission is unchanged.
2. **Agents have no internet access.** Their shell commands run under a
   seccomp filter that refuses IPv4/IPv6 sockets, and opencode's web tools are
   denied (`src/rsa/loop/no_network.py`). opencode's own connection to the
   Gemini API is unaffected. In run 1 an agent read the lab's pragmods RSA
   code from GitHub.
3. **Recovery verdict on RMSE alone** (≤ 0.01 to the ground truth). The
   held-out clause passed the seeds on the salience data; it is now only
   reported, beside the seeds' and the closest live model's distances.

Everything else follows `HANDOFF_sherlock_run1.md`; this file lists only what differs.
Run 1's environment fixes are now defaults, so most of its §2 exports are gone.

## 0. Before submitting

- **Code:** `auto-rsa` at or after the commit that adds this file. Pull in
  `~/auto-psych` (git 1.8: `git fetch origin auto-rsa && git checkout auto-rsa
  && git merge --ff-only origin/auto-rsa`).
- **Do not commit on Sherlock.** Code changes are made in the cloud session; a
  fix found on Sherlock goes back as a note (or, if it blocks the run, as a
  commit you report in the results file, as 19ae7f2 and 12d4b20 were).
- **Defaults that replace run 1's exports** (`scripts/rsa/slurm/_env.sh`,
  `submit.sh`; override any of them by exporting it):

  | setting | default now | run 1 exported |
  |---|---|---|
  | `WORK_ROOT` | `$SCRATCH/auto-psych/rsa_run2` | `rsa_run1c` |
  | venv | `$GROUP_HOME/venvs/auto-psych_rsa_run2` | `UV_PROJECT_ENVIRONMENT=...` |
  | `SSL_CERT_FILE` | `/etc/pki/tls/certs/ca-bundle.crt` when it exists | same, by hand |
  | `VENV_MODE` | `wheels` | `wheels`, by hand |
  | `OPENCODE_BIN_DIR` | `$GROUP_HOME/software/npm-global/bin` when it has opencode | same, by hand |
  | partition | `mcfrank` | `PARTITION=mcfrank` |
  | array size | 4 CPUs, 30G, 24:00:00 | `CPUS_PER_TASK=4 MEM=30G` |

  `--qos=long` is never added (not on this account); a limit over 48 h on
  `normal` stops `submit.sh` before anything is submitted.

## 1. Prepare (a job, not the login node)

```bash
cd ~/auto-psych
sbatch --chdir="$HOME/auto-psych" scripts/rsa/slurm/prepare.sbatch
```

The data should hash exactly as in run 1 (`data/rsa/sherlock_run1/SHA256SUMS`).
Compare the new `$WORK_ROOT/data/SHA256SUMS` and report any difference before
going on.

## 2. Submit

```bash
cd ~/auto-psych && bash scripts/rsa/slurm/submit.sh
```

## 3. Checks after round 1 of the first cell (beyond run 1's)

- **No network, verified at start:** each cell's loop log shows no
  `no-network shell ... does not run here` or `... left internet sockets open`
  error. The loop checks this on the node before any agent runs and stops
  if it fails.
- **No web use in the logs:** `agent_activity.md` (end of cell) should list
  no URLs the agents fetched. During the run,
  `grep -l '"tool":"webfetch"' results/round_1/*/agent.jsonl` may list agents
  that *tried*; their calls were denied. A completed one stops the loop with
  "the agent used the web".
- **CV ran:** `results/.cv/folds.json` exists with 5 folds, and the standing in
  `results/history.json` carries `elpd_cv`, `cv_diff` and `cv_dse` for every model.
- **Agents still pass their self-checks** (PASS/FAIL lines in the logs). The
  check needs no network.

## 4. Time and cost

- **CPU.** CV adds 5 fold fits per model, each about 1 min single-threaded:
  about 30 fits (~8 min on 4 CPUs) per round. Expect run 1's 4-7 h per cell
  plus about an hour.
- **Agents.** Run 1 spent $313 on 220 agent runs ($1.42 each). Without web
  access agents may spend fewer turns browsing.

## 5. Failure handling (additions)

| Symptom | Do |
|---|---|
| loop: `no-network shell ... does not run here` | The node refused the seccomp filter. Do not run with `--agent-network`; report the message and the node (`hostname`, `uname -r`). |
| loop: `the agent used the web although its web tools are denied` | A harness bug: report the agent's log path. Resubmitting the task resumes the cell. |
| loop: `selection must be 'cv' or 'loo'` / `folds.json records other folds` | Code or data changed under a resumed cell: report; do not delete `.cv/` by hand. |

Resubmitting an interrupted task resumes it from its last scored step (as in run 1's §5).

## 6. What to bring back

As in run 1's §6, with these changes:

- Use `WORK_ROOT` and the `sherlock` SSH alias in the rsync commands, not a
  hard-coded `rsa_run1` and `login.sherlock.stanford.edu`.
- Also bring back each cell's `results/.cv/folds.json`.
- Into `data/rsa/sherlock_run2/`. Then write `run_notes.json` as in
  `data/rsa/sherlock_run1/`: wall time and MaxRSS per cell, plus the findings
  and decisions. Build the overview page with
  `uv run python -m src.rsa.sweep_report --sweep data/rsa/sherlock_run2`.

**Write `docs/auto_rsa/SHERLOCK_RUN2_RESULTS.md`** as for run 1. Lead with:

- **Did CV selection export models that generalise?** Give the exported model's
  held-out rank in each real cell, and compare with run 1: 26/40 and 4/40.
- **Did recovery still succeed without internet access?** Give the exported
  and closest-live RMSE, and `seeds_would_pass`.

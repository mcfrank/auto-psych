# Handoff: RSA inner loop, Sherlock run 2

For the local session that runs RSA jobs on Sherlock. Run 2 has seven cells,
each running 6 agent slots per round:

- **real data:** 3 replicates × 8 rounds. Run 1's real cells were still
  improving at round 5, and its two replicates differed by about 90 lpd.
- **recovery_literal and recovery_salience:** 2 replicates × 4 rounds each.
  Run 1's recovery cells were flat after round 3.

It also makes six changes the PI decided on 2026-10-08 after run 1
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
4. **A live-set cap of 12** (was 8), ranked by total ELPD-CV.
5. **Seven cells, not six** (above). The array is
   `real_rep1-3, recovery_literal_rep1, recovery_salience_rep1,
   recovery_literal_rep2, recovery_salience_rep2` (tasks 0-6). The long real
   cells come first, so the cell that waits for a free slot on the 24-core node
   is a short one.
6. **Agents see per-source standings.** Their briefs say where each model in
   the set leads and lags by source. This is description only; no rule uses
   source labels.

These rules were fixed on 2026-10-08, before run 2 (`PLAN.md`, "Decisions
(2026-10-08)"). Run 2 tests them; do not change any of them mid-sweep. A
blocking bug is the only exception, and you report it.

Everything else follows `HANDOFF_sherlock_run1.md`; this file lists only what differs.
Run 1's environment fixes are now defaults, so most of its §2 exports are gone.

## 0. Before submitting

- **Code:** `auto-rsa` at or after the commit that adds this file. Pull in
  `~/auto-psych` (git 1.8: `git fetch origin && git checkout auto-rsa
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
  | array size | 4 CPUs, 30G, 24:00:00 (36G after run 2) | `CPUS_PER_TASK=4 MEM=30G` |

  `--qos=long` is never added (not on this account); a limit over 48 h on
  `normal` stops `submit.sh` before anything is submitted.

- **Unset run 1's `MAX_ITERATIONS`** if your shell still exports it: it
  would override every cell's rounds. The per-condition settings are
  `REAL_REPLICATES`, `REAL_ROUNDS`, `RECOVERY_REPLICATES` and
  `RECOVERY_ROUNDS` (defaults 3, 8, 2, 4; `scripts/rsa/slurm/_cells.sh`).
  Export them in the shell that prepares and submits, if the PI changes them.

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
  `results/history.json` carries `elpd_cv`, `cv_diff`, `cv_dse` and
  `cv_behind_by_source` for every model.
- **Per-source standings reached the agents:** a round-2 `CONTEXT.md` or
  `existing_hypotheses.md` lists models with "by source: ...".
- **Agents still pass their self-checks** (PASS/FAIL lines in the logs). The
  check needs no network.

## 4. Time and cost

- **Wall time.**
  - Run 1's real cells took 4.5-5 h for 5 rounds. At 8 rounds, plus grouped
    CV (5 fold fits per model, about 8 min a round on 4 CPUs), expect about
    9-10 h.
  - Recovery cells at 4 rounds: about 4 h.
  - Six cells run at once (24 cores, 4 each). The seventh starts when the
    first recovery cell ends, so the sweep should end about 10-12 h after
    submission. `TIME=24:00:00` per cell leaves room.
- **Agents.** Run 1 averaged $1.42 per agent run, about $8-9 per round of six
  with retries and repairs.
  - Real: 3 × 8 rounds ≈ $210.
  - Recovery: 4 × 4 rounds ≈ $140.
  - Run 2 in all: **about $350-400** (run 1: $313).
  - Without web access, agents may spend fewer turns browsing.

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
- Into `data/rsa/sherlock_run2/`, for all seven cells (`rsa_status.sh`
  lists them). Then write `run_notes.json` as in
  `data/rsa/sherlock_run1/`: wall time and MaxRSS per cell, plus the findings
  and decisions. Build the overview page with
  `uv run python -m src.rsa.sweep_report --sweep data/rsa/sherlock_run2`.

The overview page's "By source" section reads per-source CV from each cell's
`history.json`; run 2 needs no `cv_rescore` step.

**Write `docs/auto_rsa/SHERLOCK_RUN2_RESULTS.md`** as for run 1. Lead with the
paper's claim 1: are the exported models better than the starting models on
held-out conditions within the same papers? Then:

- **How do the models improve over rounds?** For each cell, give the best
  model's ELPD-CV and its held-out lpd at every round. (The overview page draws
  this. Run 1's real cells were still improving at round 5.)
- **Did CV selection export models that generalise?** Give the exported model's
  held-out rank in each real cell, and compare with run 1: 26/40 and 4/40.
- **Did recovery still succeed without internet access?** Give the exported
  and closest-live RMSE, and `seeds_would_pass`.

## 7. Held-out pages (added 2026-10-08; revised after run 2)

The driver session reads each cell condition by condition on its held-out
conditions. Build those pages on Sherlock, where every model's fits are
already cached, in a job, before bringing the cells back:

```bash
cd ~/auto-psych && git fetch origin && git merge --ff-only origin/auto-rsa  # not `fetch origin auto-rsa`: on git 1.8 it leaves origin/auto-rsa stale
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/heldout_reports.sbatch
# a subset: CELLS="real_rep1 real_rep2" sbatch ... (same script)
```

For every finished cell, `src.rsa.heldout_report` reads the loop's
`responses.csv`, `.fit_cache` and `.cv` and writes into the cell's
`$CELLS_ROOT/<cell>/outputs/heldout/`:

- `report.html`, `report.bundle.json`: people vs the best seed, the exported model and the
  three best held-out models, one panel per held-out display;
- `unit_lpd.csv`: those models' lpd per unit, held out (`test`), in sample (`train`) and
  out of fold (`cv`).

The bring-back copies `outputs/`, so the pages come back with the cells. Run 2
needed a scratch copy of the brought-back cells; the job no longer does. A
few minutes per cell. A cache miss means a refit: the log line ends `fitted N
(not in ...)`; report any. The files hold aggregates only (choice counts per
display, lpd sums per unit). Pulling the checkout is safe while cells run: they
run the staged `harness_repo`, not `~/auto-psych`.

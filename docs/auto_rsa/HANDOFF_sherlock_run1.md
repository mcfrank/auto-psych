# Handoff: RSA inner loop, first Sherlock sweep (run 1)

**For:** a local Claude session that manages Sherlock (it has ssh and Duo; cloud
sessions cannot reach Sherlock). Read `PLAN.md` for the decisions behind this.

**The sweep.** Six cells: 3 conditions x 2 replicates. A replicate is a different
loop seed on the same data and split.

| Cell | Data the loop trains on | The loop starts without |
|---|---|---|
| `real_rep{1,2}` | the real combined data's train split | (nothing) |
| `recovery_literal_rep{1,2}` | choices simulated from `literal_listener` | `literal_listener` |
| `recovery_salience_rep{1,2}` | choices simulated from `rsa_l1_salience` | `rsa_l1_salience`, `rsa_l1_shared_prior` |

**Settings in every cell:**

- 5 rounds x 6 slots.
- NUTS: 4 chains x 1,000 warmup x 1,000 samples.
- Agents: sandboxed (bubblewrap), Gemini 3.8 Flash via opencode, 2,400 s each.

After each loop the job runs a held-out evaluation (every cell), then recovery
scoring (the recovery cells).

**Scripts.** All are in `scripts/rsa/slurm/`; the header of each explains it.

| Script | Where | What it does |
|---|---|---|
| `prepare_data.sh` | login node | builds the venv, fetches the sources, then combines and splits the data |
| `submit.sh` | login node | submits `rsa_setup.sbatch`, then `rsa_loop_array.sbatch` (array 0-5) |
| `rsa_status.sh` | login node | prints one line per cell |
| `rsa_setup.sbatch` | compute | checks the environment, stages the code once, simulates the two recovery datasets |
| `rsa_loop_array.sbatch` | compute | builds the cell's agent tree, checks it, runs the loop, then scores |

## 0. Prerequisites (once)

```bash
ssh login.sherlock.stanford.edu        # Duo; use ControlMaster so later commands don't re-prompt
cd ~/auto-psych && git fetch && git checkout auto-rsa && git pull && git log -1 --oneline
grep -c '^GOOGLE_API_KEY=' .secrets    # expect 1 (never print the key)
```

**opencode >= 1.18.35.** You need the shell tool's `timeout` parameter and
`"snapshot": false`.

```bash
ml load opencode && opencode --version
# If it is older, or the module is missing:
ml load nodejs && npm install -g --prefix "$GROUP_HOME/software/npm-global" opencode-ai@1.18.35
export OPENCODE_BIN_DIR="$GROUP_HOME/software/npm-global/bin"   # export in EVERY shell you run prepare/submit from
"$OPENCODE_BIN_DIR/opencode" --version                          # expect 1.18.35 (it must run on el7's glibc)
```

**Check Gemini through opencode** (from `/tmp`, so it roots nowhere important):

```bash
export GOOGLE_GENERATIVE_AI_API_KEY="$(grep '^GOOGLE_API_KEY=' ~/auto-psych/.secrets | cut -d= -f2-)"
cd /tmp && opencode run -m google/gemini-3.8-flash "Reply with the word ok."; cd ~/auto-psych   # expect: ok
```

**JAX 0.7.0 on glibc 2.17.**

- `prepare_data.sh` imports jax on the login node, and `rsa_setup.sbatch` does
  so again on a compute node.
- If the jaxlib wheel will not install or import, stop and report. `PLAN.md`
  names an Apptainer image as the fallback.

**The loop seed.**

- The replicates differ by loop seed: replicate r runs with
  `--seed BASE_LOOP_SEED + r - 1`, which seeds every NUTS fit (added
  2026-10-07). The scripts check the CLI has it and stop otherwise;
  `ALLOW_SHARED_LOOP_SEED=1` is only for an older checkout and is not needed.

Check after `prepare_data.sh`:

```bash
cd ~/auto-psych && "$SCRATCH/auto-psych/rsa_run1/venv/bin/python" -m src.rsa.loop.run --help | grep -E -- '--seed '   # a line: the CLI has it
```

**Recovery data.** The simulated files keep only the trials the loop fits
(included forced-choice trials with a display); production trials and
excluded participants are dropped so no real response sits beside simulated
ones. Setup checks those rows equal the real split's fitted rows outside
`choice`.

**Resources.**

- The array's defaults are placeholders: 16 CPUs, 64G, 48:00:00. The setup
  job's are 16 CPUs, 32G, 06:00:00.
- Before submitting, set `CPUS_PER_TASK`, `MEM`, `TIME` and `SETUP_TIME` from
  `docs/auto_rsa/SMOKE_RESULTS_2.md` (branch `auto-rsa-smoke2`), using the
  formula in section 4.
- `TIME` over 48 h adds `--qos=long` automatically (at most 7 days).

## 1. Prepare the data (login node, ~10-20 min)

```bash
cd ~/auto-psych
bash scripts/rsa/slurm/prepare_data.sh 2>&1 | tee ~/rsa_run1_prepare.log
```

It writes `$SCRATCH/auto-psych/rsa_run1/` (`WORK_ROOT`). Expect, in order:

1. `[rsa_env] opencode 1.18.x ...`
2. `[prepare] venv OK: python 3.12.x, jax 0.7.0, numpyro 0.19.0, pymc 5.28.5, ...`
3. `<source>: wrote ...` for each of the four ingested sources. The ingest
   checks every download's sha256. The committed CSVs must rebuild
   byte-identically: if the checkout changes, the script stops and shows
   `git status`.
4. `wrote .../data/combined_trials.csv: N rows`, with a per-source table.
5. One line per source: `test holds k/n units, t/T trials`. Each source's test
   share should be at least 20%, and close to it.
6. The `SHA256SUMS` lines. **Record them in your report.**

If `uv sync` fails on a missing glibc-2.17 wheel, rerun with `VENV_MODE=wheels`
(the subjective-randomness recipe: wheels only, the RSA stack pinned).

## 2. Submit (login node)

```bash
export CPUS_PER_TASK=16 MEM=64G TIME=48:00:00 SETUP_TIME=06:00:00   # from SMOKE_RESULTS_2.md
# (ALLOW_SHARED_LOOP_SEED is not needed: the loop CLI takes --seed)
bash scripts/rsa/slurm/submit.sh
```

Expect `submitted setup: <id>` and `submitted array: <id>`. Logs go to
`$WORK_ROOT/logs/rsa_setup_<id>.out` and `rsa_loop_<array>_<task>.out`.

**The setup job.** Its log should show:

- `venv OK on sh...`;
- `[stage] staged code <commit>`;
- `import chain OK`;
- for each recovery condition: `simulating`, then `simulated`, then
  `same held-out units and rows as the real split`;
- the recovery datasets' sha256s, then `[setup] done`.

The simulation logs are `$WORK_ROOT/data/simulate_<condition>.log`.

**Each array task.** Its log should show:

- `[scrub] removed seed ...` (recovery cells only);
- `[check] withheld seed ...: no file, no manifest entry, no copy`;
- `[check] test.csv: absent from the tree`;
- `[check] agent tree OK`;
- the loop's own output: seed fits, then rounds, then `best model: ...`;
- the held-out scoring, then the recovery scoring;
- `<cell> done`.

## 3. Monitor

```bash
bash ~/auto-psych/scripts/rsa/slurm/rsa_status.sh     # one line per cell
squeue -u $USER
sacct -j <array_id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -f $SCRATCH/auto-psych/rsa_run1/logs/rsa_loop_<array>_<task>.out
```

`rsa_status.sh` prints, per cell:

- the state: pending, running, done, failed or stopped;
- the stage: seed fits, round k/5, scoring, scored;
- the current best model;
- the ledger counts, as admitted/rejected/pruned;
- tokens and cost from `token_usage.jsonl` (`+` means some calls reported no
  usage);
- the recovery verdict.

To look at a running cell's report, copy it to your machine:
`$WORK_ROOT/cells/<cell>/results/report.html`.

**Check after round 1 of the first cell:**

- agents read PASS/FAIL lines from their self-checks (`grep -l "PASS\|FAIL"
  results/round_1/*/agent.jsonl`);
- no `.xdg_data/opencode/snapshot` directories;
- `MaxRSS` is well under `MEM`.

## 4. Cost and time

Fill these in from `SMOKE_RESULTS_2.md`; if it is not there yet, compute them
from its run directory.

- **Agent runs.** 6 cells x 5 rounds x 6 slots = 180 first attempts. Retries
  and repairs add up to as many again: **180-360 agent runs**.
- **Agent cost.** Multiply the runs by the per-agent cost of Gemini 3.8 Flash
  from `SMOKE_RESULTS_2.md`'s `token_usage.jsonl`. For reference, smoke test 1
  ran Gemini 3.1 Pro at $0.18-0.36 per agent ($0.71 for a 3-slot round).
  Flash should be well under that. **TODO: fill in.**
- **Wall time per cell:**

  ```
  5 x T_seed_fit
  + 5 rounds x (agent phase <= 3 x 40 min [first try, retry, repair]
                + up to 12 x T_admission_fit, run one at a time)
  + held-out scoring (cache hits)
  + one ground-truth fit (recovery cells)
  ```

  T_seed_fit and T_admission_fit are smoke test 2's times on the combined
  data, scaled from its 2 x 500 + 500 to 4 x 1000 + 1000 (about x2-4).
  Setup is about 1.2 x the slower ground-truth fit. **TODO: fill in; set
  `TIME` to about 1.5 x this.**
- **Disk per cell.** The fit cache dominates (one `.nc` per fitted model).
  Scale smoke test 2's run directory size by the number of models
  (5 seeds + up to 30-60 candidates). **TODO.** Inodes: each agent tree is
  a copy of `src/` plus the agents' directories.

## 5. Failure handling

A cell never resumes on other code. Each records its code in
`cells/<cell>/code_commit`; the sweep's code is in `$WORK_ROOT/code_commit`.

| Symptom | Do |
|---|---|
| prepare: `the ingest changed the checkout` | A source changed upstream or the ingest is not deterministic. Do not proceed; restore with `git checkout -- src/pipelines/outer_loop/projects/rsa_reference/data` and report. |
| prepare: `opencode ... is older` / `no opencode` | Install it as in section 0 and export `OPENCODE_BIN_DIR`. |
| prepare: `FATAL: bwrap not on PATH` (login node lacks the module) | Run prepare in `sh_dev -c 2 -t 1:00:00` (compute nodes have internet). |
| setup: `prepared by code X, but REPO is at Y` | Re-run `prepare_data.sh` on the current checkout, then submit again. |
| setup: `takes no --seed` | The checkout predates 2026-10-07: pull `auto-rsa` and re-run `prepare_data.sh`. |
| setup: `the simulated split holds out other units` or `differs ... outside the choice column` | A bug in simulate/split. Do not run the array; report. |
| setup: `ground truth's fit did not converge` | Report the log (`data/simulate_<cond>.log`). |
| array: `ERROR (agent tree ...)` from `check_agent_tree.sh` | A leak was caught before any agent ran. Do not override; report the listed paths. |
| array: `the loop exited` non-zero, or TIMEOUT / OUT_OF_MEMORY mid-loop | The loop cannot resume mid-run. Resubmit only that task with `RESTART=1` (the old results go to `cells/<cell>/attempts/`, and the fit cache is reused). Raise `TIME`/`MEM` if the job hit its limit: `ARRAY=<task> RESTART=1 SKIP_SETUP=1 TIME=... MEM=... bash scripts/rsa/slurm/submit.sh` |
| array: failed in held-out or recovery scoring (`export.json` exists) | Fix the cause and resubmit the task without `RESTART`: it skips the loop and resumes at scoring. |
| status `stopped` | A duplicate job exited, or the cell was already done. Check `cells/<cell>/last_exit` and the log. |
| Every slot `no file`; logs show `[USAGE LIMIT]` or quota errors | Check the Gemini quota and billing on the key, then restart the cell. |
| `$SCRATCH` quota (Errno 122) | `sh_quota`. Delete old run trees, not this sweep's `cells/` or `agent_trees/`. |

**Never:**

- point `WORK_ROOT` at a directory another sweep used;
- re-run `prepare_data.sh` once `cells/` exists (it refuses);
- edit files under `harness_repo/` or `agent_src/`.

To run on new code, use a new `WORK_ROOT` (`export WORK_ROOT=$SCRATCH/auto-psych/rsa_run1b`).

## 6. What to bring back

Per cell, into `data/rsa/sherlock_run1/<cell>/` on branch `auto-rsa`, committed
(not to `main`; no pull request). Run from your local checkout:

```bash
W=/scratch/users/<sunet>/auto-psych/rsa_run1
for cell in real_rep1 real_rep2 recovery_literal_rep1 recovery_literal_rep2 recovery_salience_rep1 recovery_salience_rep2; do
  mkdir -p data/rsa/sherlock_run1/$cell
  rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='/report.html' --include='/report.bundle.json' --include='/history.json' \
    --include='/attempted_hypotheses.jsonl' --include='/export.json' --include='/best_model.py' \
    --include='/token_usage.jsonl' --include='/token_usage_summary.json' --include='/novelty_pool.json' \
    --include='/models/***' \
    --include='/heldout/heldout.csv' --include='/heldout/heldout.json' \
    --include='/recovery/recovery.json' --include='/recovery/heldout.csv' --include='/recovery/heldout.json' \
    --include='/round_*/*/hypothesis.md' --include='/round_*/*/model_name.txt' --include='/round_*/*/candidate.py' \
    --exclude='*' \
    login.sherlock.stanford.edu:$W/cells/$cell/outputs/ data/rsa/sherlock_run1/$cell/
  rsync -av login.sherlock.stanford.edu:$W/cells/$cell/{cell.json,code_commit,data.sha256,agent_activity.md} data/rsa/sherlock_run1/$cell/
  rsync -av login.sherlock.stanford.edu:$W/cells/$cell/gt_name_mentions.txt data/rsa/sherlock_run1/$cell/ 2>/dev/null || true
done
rsync -av login.sherlock.stanford.edu:$W/data/{SHA256SUMS,SHA256SUMS.recovery,prepared_code} data/rsa/sherlock_run1/
rsync -av login.sherlock.stanford.edu:$W/data/real/split.json data/rsa/sherlock_run1/split_real.json
for c in recovery_literal recovery_salience; do
  rsync -av login.sherlock.stanford.edu:$W/data/$c/provenance.json data/rsa/sherlock_run1/provenance_$c.json
done
```

**Committing summaries is fine** (PI decision 2026-10-07): `report.bundle.json`,
`report.html` and the held-out/recovery tables hold aggregate numbers, and those
are fine to commit even where they come from sources without a licence.

**Do not commit:**

- trial-level data: `responses.csv`, `train.csv`/`test.csv`, simulated CSVs,
  `*pointwise*.csv`. The unlicensed and CC BY-NC-ND sources are derived at
  run time, never committed.
- `.fit_cache/`;
- agent logs (`agent.jsonl`, `scratch/`): agents print data rows into them.

Leave those in `$WORK_ROOT/cells/<cell>/outputs/` on Sherlock.

**Write `docs/auto_rsa/SHERLOCK_RUN1_RESULTS.md`** with:

- the commit, `SHA256SUMS` and job ids;
- `rsa_status.sh` at the end;
- per cell:
  - best model and its held-out lpd versus the best seed;
  - the admitted/rejected/pruned counts and the main rejection reasons;
  - the recovery verdict;
  - tokens and cost;
  - wall time and MaxRSS (`sacct`);
- anything in `agent_activity.md` or `gt_name_mentions.txt`;
- every failure and what you did.

**Cleanup.** Only once the PI has looked: the agent trees (inodes and the fit
cache) are at `$SCRATCH/auto-psych/agent_trees/$(cat cells/<cell>/agent_tree_id)`.

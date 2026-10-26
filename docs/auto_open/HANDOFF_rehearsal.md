# Handoff: the open campaign's rehearsals on Sherlock

From the driver session, 2026-10-21, for the local Sherlock session. Branch `auto-psych-open`.
No money is spent: the experiments' participants are simulated. The agents (the designer, the
theorists) and the pilot's participants are real Gemini calls, so there is API spend.

The open campaign (`docs/auto_open/PLAN.md`) lets a designer agent make a new paradigm every
experiment. Theories must predict every paradigm with one set of parameters. Two runs, in order:

- **A, the dry run (the last piece of the Phase 3 gate):** chain 0, 1 experiment, 2 inner-loop
  rounds.
- **B, the rehearsal (Phase 3):** chain 0, 3 experiments, 5 rounds each, in a fresh work root.

Start B only after the driver has reviewed A.

## Restart, 2026-10-22 (answers `DRY_RUN_STATUS.md`)

The stalled fit (`rsa_l1_shared_prior`) was the link's temperature trading off against the
theory's own sharpness on E. The PI decided (PLAN.md, Decisions 2026-10-22):
- **E's link temperature is fixed at 1.** Its lapse and every new paradigm's temperature stay free.
- **Design fits** take the previous loop's fit, else a cheaper fit within 2 h. A theory whose fit
  fails or runs out of time is left out of that design, recorded in
  `experiment<n>/design_screened_out.json`.

Locally, on pragmods, the stalled model's two-chain fit takes 77 s with the temperature fixed.

1. Cancel 48820332_0 and pull `auto-psych-open`.
2. **Delete the ground truth:** `rm -r $SCRATCH/auto-psych/rsa_open_dry2/private/ground_truth`.
   It was fitted with E's temperature free, so its parameters sit on that ridge (`alpha` small,
   temperature about 2.5). The job refits it under the new convention (15–60 min).
3. **Restart run A, trimmed** (PI 2026-10-22: run A tests the machinery with real agents; run B
   is the rehearsal). It fits the pragmods trials alone (6,703 of E's ~49k), runs one round of 3
   theorists (one exploratory slot, one refining the incumbent, one refining the agent's pick; the
   critic first), a 4-person pilot and 3 CV folds. The designer keeps its full budget, since it is
   the part we most need to see. Expect about 4–6 h, mostly the designer:

   ```bash
   cd ~/auto-psych && git pull
   export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_dry2
   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" --time=12:00:00 \
     --export=ALL,WORK_ROOT=$WORK_ROOT,RESTAGE_REASON='E reference block, design-fit limit, trimmed run A; PI 2026-10-22',LITERATURE=pragmods,N_EXPERIMENTS=1,MAX_ITERATIONS=1,CANDIDATES=3,PILOT_PARTICIPANTS=4,CV_FOLDS=3 \
     scripts/rsa/slurm/open_rehearsal.sbatch
   ```

   Every fit is redone: the corpus changed. Its ground truth is then fitted on pragmods alone, so
   **run B must not copy it** (§3 now says so).
4. **Submit the audit alongside it.** It runs no agents and spends no API money:

   ```bash
   mkdir -p "$SCRATCH/auto-psych/rsa_open_audit/logs"
   sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_open_audit/logs/%x_%j.out" \
     scripts/rsa/slurm/open_temperature_audit.sbatch
   ```

   It fits all 15 bar theories on the full E with the temperature fixed and free, and reports
   each one's fit time, divergences, temperature and the difference in held-out fit (PSIS-LOO,
   with an SE clustered by display). Bring back `temperature_audit/audit.{json,md}` and the log
   to `data/rsa_open/temperature_audit/`.

## 0. What the job does (`scripts/rsa/slurm/open_rehearsal.sbatch`)

1. **Once per work root:**
   - stages the code (`stage_code.sh`, as the closed sweeps do);
   - copies the ported seeds (`data/rsa_open/seeds`: the bar and each chain's starting set);
   - builds the hidden ground truth into `$WORK_ROOT/private/ground_truth/`
     (`src.rsa_open.ground_truth`).

   The truth is the closed rehearsals' coherent truth, ported to a theory and fitted once to E,
   then fixed at its posterior means. E is all literature trials, run 2's
   `promote_work/all_trials.csv`. This fit takes roughly 15–60 min.
2. **Builds the chain's agent tree,** then checks it before any agent starts:
   - `check_agent_tree.sh`;
   - no file outside `_runs/` names the truth;
   - no copy of the ported truth anywhere.
3. **Runs `src.rsa_open.run_chain`.** Each experiment has three stages.
   - **Design (about 6 h at most).** Up to two designer cycles. Each cycle is up to 2 h of agent
     time, plus the harness's checks of each submission: EIG about 12 min (about 3 min from run
     B on), the link check about 7 min, then a 12-person simulated pilot with its legibility
     check.
   - **Collection.** The simulated people answer every list of the frozen paradigm from the
     truth.
   - **The inner loop.** Six theorists per round.
4. **Copies the run** (without fit caches) to `$CELLS_ROOT/open_chain_0/outputs`, and lists in
   `gt_name_mentions.txt` any agent file that names the truth.

Resubmitting resumes. Finished stages are skipped. An interrupted design or loop restarts: its
attempt is kept as `design.abandoned_<i>` / `loop.abandoned_<i>`, and a loop's fit cache carries
over.

## 1. Pull and check prerequisites

```bash
cd ~/auto-psych && git fetch origin && git checkout auto-psych-open && git merge --ff-only origin/auto-psych-open
ls "$SCRATCH/auto-psych/rsa_run2/promote_work/all_trials.csv"   # E (from promote.sbatch)
grep -n gemini-4-argon opencode.json                             # the designer's model is declared
```

Two more prerequisites:

- `.secrets` has `GOOGLE_API_KEY`; `_env.sh` derives opencode's `GOOGLE_GENERATIVE_AI_API_KEY`
  from it.
- opencode is at least 1.18.35 (`_env.sh` checks this).

If run 2's root is elsewhere, pass `RUN2_ROOT=<its root>`.

**The shared venv needs the open campaign's packages** (google-genai, pydantic, resvg-py; the
first submission failed without them). Once, on a login node:

```bash
source scripts/rsa/slurm/_env.sh && source scripts/_venv.sh && build_shared_venv
(cd ~/auto-psych && "$VENV_PY" -m src.rsa_open.env_check)   # should print "[env_check] OK"
```

The job runs the same check first and stops before any staging, fit or agent if it fails.

## 2. Run A: the dry run

```bash
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_dry
mkdir -p "$WORK_ROOT/logs"
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" --time=1-00:00:00 \
  --export=ALL,WORK_ROOT=$WORK_ROOT,N_EXPERIMENTS=1,MAX_ITERATIONS=2 \
  scripts/rsa/slurm/open_rehearsal.sbatch
```

Expect about 6–14 h: the design, then 2 loop rounds.

**What to look at:**

- The job log's header lines: what was withheld from the agents, and the chain line.
- `$WORK_ROOT/private/ground_truth/ground_truth.json`: the truth's fit on E, and its
  parameters.
- `outputs/experiment1/design/`:
  - every `cycle_*/logs/brief_*.md`;
  - the tool log (`cycle_*/work/tool_log.jsonl`);
  - each submission's checks (`cycle_*/checks/<n>/`): the eig, the link check, and
    `pilot/pilot.json` with its legibility block;
  - `frozen/` (the paradigm, the lists, `record.json`).
- `outputs/experiment1/data/block.json` (the simulated responses), and
  `outputs/experiment1/loop/` (`history.json`, `attempted_hypotheses.jsonl`, `export.json`,
  each round's candidates).
- `gt_name_mentions.txt`: should be empty.
- `token_usage` records: what Gemini 4, Gemini 3.8 Flash and the pilot cost.

**Bring back** to the driver, under `data/rsa_open/dry/` on `auto-psych-open`:

- `outputs/` without fit caches (it is all simulated, with no participant ids);
- `private/ground_truth/ground_truth.json`;
- the job log.

Do not bring back `private/ground_truth/ground_truth.py` under any name the agents' trees could
pick up later. `data/` is excluded from every agent tree, so `data/rsa_open/dry/` is safe.

## Run B: let it finish; bring back experiments 1 and 2 now (revised 2026-10-24)

Answers the Sat 09:16 update of `RUN_B_STATUS.md`. The 2026-10-23 request (bring back
experiment 1, then cancel) never reached Sherlock, and run B is now in experiment 3's design. So:

1. **Leave run B running to its end.** It has a day or so left, and its full three experiments
   on the old code are the baseline the fixed code is compared against.
2. **Now, without stopping the job,** copy experiments 1 and 2 to `data/rsa_open/rehearsal_b/`
   on `auto-psych-open`, so the driver can profile the too-slow candidates before the run ends:
   - `chain_0/experiment{1,2}/loop/`: every `round_*/` directory, with each candidate's
     `candidate.py`, `hypothesis.md`, `model_name.txt`, its `*_retry_*`/`*_repair_*` siblings,
     and `agent.jsonl`; leave out `scratch/` and `.xdg_data/`;
   - `attempted_hypotheses.jsonl`, `history.json`, `models/` (with `pruned/`) and the
     `.fit_cache/*.json` sidecars, not the `.nc` files;
   - `chain_0/experiment{1,2}/design/frozen/`, `design_screened_out.json`, and the job log so far.
3. **At the end,** bring back everything per §3 into the same directory.

## The ground-truth check on run B′ (2026-10-26, PI)

**What:** score run B′'s hidden ground truth on run B′'s own corpus and folds, beside every theory
the loop ended with or pruned (`src/rsa_open/oracle.py`). Only the truth is fitted; the others come
from the replay's fit cache. No agents, no API spend. Expect 1–3 h on the node's 24 cores.

```bash
cd ~/auto-psych && git pull
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_replay_b1
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,WORK_ROOT=$WORK_ROOT scripts/rsa/slurm/open_oracle.sbatch
```

The log should show the first fits as cache hits (no sampling time) and one new fit for
`ground_truth` and its folds. **Bring back** `$WORK_ROOT/private/oracle/{oracle.json,oracle.md}`
and the job log to `data/rsa_open/replay_b1/oracle/`. They name the truth only as
`ground_truth`, and hold aggregates only.

## Run B′: replay run B's experiment 1 on the fixed code (2026-10-24)

> **Hold: launch once run B has finished** (it needs the node's 24 cores) **and the PI has said go.**

Run B′ reruns **only the inner loop of run B's experiment 1**: same frozen design, same simulated
data, same starting theories, new code. The difference from run B's experiment 1 (38 attempts, 24
too slow, 9 admitted) is then the code's alone. The new code changes:
1. **The admission limit is corpus-relative** (PI 2026-10-24): 2× the median full-fit time of the
   loop's starting theories, never below 30 minutes. That's about 90 minutes on all of E, where
   4 of chain 0's 6 starting theories take longer than 30 minutes themselves. The loop prints
   the limit and records it in `history.json` (`admission_limit_sec`) and in every theorist's
   `CONTEXT.md`.
2. **The self-check reports the sampler**: steps per draw, step size, parameter pairs correlated
   above 0.85, and a projected admission-fit time (within about 40% of real full fits on
   pragmods). A projection over the loop's limit is a FAIL. Rejections carry the same report.
3. **The theorists' prompt** names RSA's identification pairs and no longer invites a `beta`.
4. **Near-miss refits only; each fold limited at 3× its theory's full fit** (a failed fold:
   untrusted, `cv_problem` in `history.json`).

**A fresh work root**, with run B's ground truth copied in. `REPLAY_FROM` copies run B's
experiment 1, everything but its loop, into the new run. The job refuses if the two ground truths
differ. The loop fits its starting theories afresh, because run B's cached fits do not record
their time and the limit needs it.

```bash
cd ~/auto-psych && git pull
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_replay_b1
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/private"
cp -r "$SCRATCH/auto-psych/rsa_open_rehearsal/private/ground_truth" "$WORK_ROOT/private/"
export RESTAGE_REASON='Replay of run B experiment 1: corpus-relative admission limit, sampler diagnostics, identification prompt, near-miss refits, bounded folds; PI 2026-10-24'
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" --time=2-00:00:00 \
  --export=ALL,WORK_ROOT=$WORK_ROOT,REPLAY_FROM=$SCRATCH/auto-psych/rsa_open_rehearsal,N_EXPERIMENTS=1,MAX_ITERATIONS=3,CANDIDATES=6 \
  scripts/rsa/slurm/open_rehearsal.sbatch
```

The log should show `replay: experiment 1's design and data from ...`, then no designer, then
`[open loop] admission limit N min: ...`.

**Expect 16–24 h and $30–40.** The starting fits and their folds take about 4–5 h; the slowest
theory takes 2.7 h per fit on all of E. Then 3 rounds of 4–6 h each, longer than run B's because
the limit is longer. Run B's theorists cost about $1.2 a session; this is about 30 sessions.

**What to report:** the admission limit; per round, attempts / admitted / too slow / not
converged; whether the rejection reasons and the theorists' self-check output show the sampling
lines; any `cv_problem`; the incumbent per round. Run B's experiment 1 gives the same numbers for
comparison.

**Bring back** to `data/rsa_open/replay_b1/`, as §3 says for run B.

## Redeploy run B with the faster fitting (2026-10-22, PI)

> **Cleared (2026-10-22).** Warm-started folds give the cold folds' held-out scores. On pragmods
> with 3 folds: `rsa_l1_salience` −5,081.5 against −5,083.8, `rsa_l2` −5,005.9 against −5,006.1,
> all converged, in 372 and 385 s against 569 and 655 s. On all of E sampling dominates the fixed
> compile cost, so the saving should be nearer 2.5×. Redeploy now.

Three speedups, none of which changes what is fitted or admitted (PLAN.md, Decisions 2026-10-22):
1. **Cross-validation folds start at their theory's full fit** (posterior medians) and run 300
   warm-up + 500 draws per chain instead of 1,000 + 1,000. The folds were about 85% of the loop's
   fitting. The full fits, the admission fits and the 30-minute admission limit are unchanged.
2. **24 cores** for the job, not 16. It needs the whole `mcfrank` node: check nothing else is
   running there.
3. **The theorists' self-check** fits a fixed fifth of E's displays. Admission still fits all of
   E.

Expect roughly 1–1.5 days instead of 2–3. That's an estimate, not a measurement.

**Redeploy in the same work root.** What run B has done so far is kept:
- the ground truth (`private/ground_truth`) and every full fit keep their cache keys, so nothing
  is refitted;
- a frozen design is kept; an unfinished one restarts, its attempt kept as `design.abandoned_<i>`;
- an unfinished loop restarts, its attempt kept as `loop.abandoned_<i>` and its fit cache carried
  over, so only the fold fits are redone.

```bash
scancel <run B's job id>
cd ~/auto-psych && git pull
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_rehearsal
export RESTAGE_REASON='Faster fitting: warm-started CV folds, 24 cores, self-check on a fifth of E; PI 2026-10-22'
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" \
  --export=ALL,WORK_ROOT=$WORK_ROOT scripts/rsa/slurm/open_rehearsal.sbatch
```

Set `RESTAGE_REASON` in the environment, not in `--export=`: its commas would split it. Everything
else is as in §3.

## 3. Run B: the rehearsal (reviewed and cleared, 2026-10-22)

The driver reviewed run A (`DRY_RUN_A_REPORT.md`) and the temperature audit. Both are clear
(PLAN.md, phases table). The code run B stages adds two things run A's report asked for:
- **Spend:** the chain writes every agent's and the pilot's spend to
  `chain_0/token_usage.jsonl` and `token_usage_summary.json`.
- **Restarts:** a restart after a data change sets the stale corpus aside instead of failing (§7
  of the report).

Run B's fresh work root stages the branch as it is then. Unlike the trimmed run A, it fits all of
E (its own ground truth included), with the full rounds, theorists, pilot and folds.

```bash
cd ~/auto-psych && git pull
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_rehearsal
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/private"
# No ground truth copied: run A's was fitted on pragmods alone; run B fits its own on all of E.
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" \
  --export=ALL,WORK_ROOT=$WORK_ROOT scripts/rsa/slurm/open_rehearsal.sbatch
```

The defaults are 3 experiments, 5 rounds of 6 theorists, a 12-person pilot, 5 CV folds, all of E
(`LITERATURE=all`), and a 5-day limit on `mcfrank`.

**Expect about 2–3 days.** These are estimates from run A and the audit, not measurements. Per
experiment:
- the truth fit: once, 0.5–1 h;
- the design fits: about 1 h; the slowest bar theory is about 50 min at the design's settings;
- the designer and its checks: 1–6 h;
- the loop's starting fits and their 5 folds on all of E: 3–6 h. The slowest starting theory,
  `isolated_graded_costly_l3`, took 2.7 h per fit in the audit;
- 5 rounds: 2–3 h each, plus the critic.

**Spend:** run A's one round of 3 cost $9.28 for 5 agent sessions, roughly $1.3–1.7 a session.
Run B has 3 designer sessions (or up to 6 with retries) and 3 × 5 × (6 theorists + critic) ≈ 105
loop sessions, so on the order of $150–250 plus the pilot. The chain's
`token_usage_summary.json` now records it.

**What to watch** (none blocks the run):
- `experiment<n>/design_screened_out.json`: a bar theory left out of a design;
- the loop's `attempted_hypotheses.jsonl`: candidates rejected as too slow. The admission fit's
  30-minute limit is the closed campaign's. On all of E it may reject refinements of the heaviest
  theories, as the closed campaign's log noted; that's parity, but count them;
- the critic's `critiques.md` in each round;
- `gt_name_mentions.txt`, which should be empty.

**Bring back** to `data/rsa_open/rehearsal/`, as for A:
- `outputs/` without caches;
- `private/ground_truth/ground_truth.json`, but not `ground_truth.py`;
- `gt_name_mentions.txt`, `code_history` and the job log;
- the chain's `token_usage_summary.json`.

## 4. If something fails

- **A designer cycle that fails twice stops the chain** (`DesignerFailed`, with each cycle's
  reason). This is the plan's rule ("fail loudly, no fallback design"). Bring back
  `design/cycle_*` and the job log, and do not resubmit until the driver has read them.
- **An agent's usage limit** waits and retries (`src/runtime/usage_limits.py`), up to 12 h.
- **The job's time limit:** resubmit the same command. The chain resumes.

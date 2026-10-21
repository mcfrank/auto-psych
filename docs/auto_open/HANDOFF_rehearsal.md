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

## 3. Run B: the rehearsal (after the driver's review of A)

Run B's fresh work root stages the branch as it is then. That includes what was built after run
A was submitted, all of which A lacks:

- the critic before every inner-loop round;
- the lazy EIG search (about 6× faster);
- a pilot that sends only the current screen's picture (about a tenth of the tokens);
- bets in the EIG tool.

```bash
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_rehearsal
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/private"
cp -r "$SCRATCH/auto-psych/rsa_open_dry/private/ground_truth" "$WORK_ROOT/private/"   # same truth, no refit
sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%A_%a.out" \
  --export=ALL,WORK_ROOT=$WORK_ROOT scripts/rsa/slurm/open_rehearsal.sbatch
```

The defaults are 3 experiments, 5 rounds of 6 theorists, a 12-person pilot, and a 5-day limit
on the `mcfrank` partition. Bring the run back to `data/rsa_open/rehearsal/`, as for A.

## 4. If something fails

- **A designer cycle that fails twice stops the chain** (`DesignerFailed`, with each cycle's
  reason). This is the plan's rule ("fail loudly, no fallback design"). Bring back
  `design/cycle_*` and the job log, and do not resubmit until the driver has read them.
- **An agent's usage limit** waits and retries (`src/runtime/usage_limits.py`), up to 12 h.
- **The job's time limit:** resubmit the same command. The chain resumes.

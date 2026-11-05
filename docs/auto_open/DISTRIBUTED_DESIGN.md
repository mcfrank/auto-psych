# Design note: running stage 2 across the cluster (v3, 2026-11-03)

**Status: a proposal; nothing of §0a is built.** v1 answered the PI's question of 2026-11-02; v2
took in the Sherlock session's review (`DISTRIBUTED_DESIGN_REVIEW.md`); **v3 follows SRCC's answer**
(§0a), which replaces v1's and v2's long-running driver. What v2 built stays (§0, "Built first").
Open chain 1 is not affected: it finishes on its tag.

## 0a. SRCC's answer, and the design it points to

The PI asked SRCC (ticket #106983, 2026-11-03) the review's three questions. Mark Yoder's reply,
in short:
- **It is within Sherlock's policies**, "especially since you are mostly using your own
  partition", including **a job that submits other jobs**, with a mechanism so it can't run away
  (Sherlock's QoS limits on jobs per user and group protect the system, but a workflow that hits
  them can break).
- **Infrastructure failures are Slurm's:** a failed node is drained and reset, and its jobs are
  requeued. Keep an eye on it, and on whether our jobs contribute.
- **Arrays of single-process jobs** for the fits, throttled (`--array=1-100%20`), each at least
  about 20 minutes (ours are 40-80). **Not** a large allocation running many independent fits
  (inflexible, under-used), **not** `sbatch --wait` (leaves resources idle).
- **No orchestrator job alongside**: "that usually does not work out well". Orchestration goes in
  the steps themselves: each piece of work writes its own output, and **a follow-up job gathers
  the results and launches the next step** ("aggregating the output from the previous ensemble is
  part of the setup for the next").
- "Don't try to invent a scheduler; use the one we have." Start small and simple, then take small
  steps. (Sherlock office hours: Tuesdays 10:00, Thursdays 15:00.)

**So a chain becomes a sequence of Slurm steps, each launched by the last; nothing waits.**

```
 step job (small, minutes)                       arrays (owners/normal, -C CPU_GEN:SIE for fits)
 ─────────────────────────                       ──────────────────────────────────────────────
 loads the chain's state from disk
 runs until it needs outside work:
   fits not in the cache    ── sbatch --array ──▶ fit tasks: 1 core, --mem from the theory's peak,
   agent sessions not done  ── sbatch --array ──▶ agent tasks: one sandboxed session each (60 min)
 submits its own continuation
   --dependency=afterany:<the arrays>
 and exits                                       each task writes its own result file
 ─────────────────────────
 the continuation: same code, finds the work done (cache hits, agents' outputs), goes on
```

- **The step** is today's `run_chain` and `OpenLoop`, run in a "step mode": where it now fits in a
  child (`_fit_in_child`, the backend seam built in v2) or runs agents (`_spawn_all`), it instead
  records what's missing, submits it as an array with its own continuation, and exits. Everything
  between (admission, scoring, pruning, export, the design's checks) runs inside the step, as now.
- **Fits** are already content-addressed and cached: a fit task writes its `.nc` and sidecar (and
  its child's report: ok, the theory's failure, or infrastructure) and the continuation reads them.
  No shared HDF5: each task's own file is what Mark's "each instance writes its own output" means
  here, and needs no concurrent writes.
- **Agent sessions** get a done-marker (exit status, time) beside their outputs, so a continuation
  never runs a finished session again. Theorists, critics, retries and repairs become arrays; the
  designer and its EIG search run in one design step (about 8-24 cores, 2-3 h).
- **Checkpoints:** the steps that change state (an admission pass, a scored step, the end of a loop)
  write it to disk before going on: the live set and ledger already are; the round's events and
  each phase's completion are added (`round_<k>/phases.json`). A continuation resumes after the
  last checkpoint, extending Addendum B's resume from "after a scored round" to "after a phase".
- **Collection** is one 1-core job per study (up to 24 h; it waits on Prolific, it doesn't compute).
- **Runaway guards:** every step checks the cell's STOP before submitting; a per-experiment budget
  of steps (say 60) and of resubmissions per fit (2 after an infrastructure failure, 1 with twice
  the memory after an out-of-memory kill); arrays throttled (`%24` per chain); and a step refuses
  to submit when the campaign already has more than a set number of jobs queued (one `squeue`
  count at submission, never a polling loop).
- **Unchanged:** every setting, seed, cache key, time limit (measured inside the fit task, as now),
  gate, admission order, score and rule. Admission stays sequential after its fits are in, as
  today's prefit-then-admit. So this is an infrastructure amendment, like Addendum B.

**What it costs:** a few minutes of scheduling between steps, perhaps 6-8 steps a round, so about
30 minutes a round on top of the about 5-6 h critical path the review measured; a chain still
about 4 days, all 7 at once; nothing idle while fits run. **To build:** the step mode with
phase checkpoints (the main work: 3-4 days), the fit and agent array scripts (1 day), then a
small-to-large rehearsal: one simulated round, then one simulated chain, then two at once (Mark's
"start small"). About a week before the rehearsal is done.

**Compute optimizations** (folds' warmup is 54% of fitting; parallel chains; saved CV scores): `DISTRIBUTED_DESIGN_REVIEW_V3.md` §9, proposed, with tests in §10.

**Superseded by §0a:** v1's file queue and workers (§3), v2's driver jobs on `normal` (§0, §4) and
v1/v2's automatic resume (Slurm requeues failed nodes' jobs; a step's continuation is the resume).
Still in force: v2's CPU restriction and machine record, clean environments for fit tasks (no
`.secrets`), and the status command and mail.

## 0b. The plan after the v3 review (PI, 2026-11-03)

The Sherlock session's review of v3 (`DISTRIBUTED_DESIGN_REVIEW_V3.md`) measured what v3 had
estimated. What it changes, and what the PI decided:

- **SRCC's answer stands.** The PI is comfortable going forward on Mark Yoder's reply (ticket
  #106983), which answered the question as asked: whether a chain's next job may "resume it and start
  more agent sessions when no one is watching". No further check is planned.
- **Agents run inside steps, not as arrays.** Theorist sessions use 3-7% of a core; arrays of them
  would be the under-use SRCC warned of. **3-4 coarse steps a round** (review §3): an agent step
  (critique, theorists, retries; it submits the first candidates' fit array), an admission step
  (admits, submits folds, runs repair agents while they fit), and one or two to admit repairs, score
  and prune. Each lasts long enough for rule 10's minimum, and nothing waits idle.
- **A restart must be cheap first (A1, built 2026-11-03, below):** a restart cost 74 min and 25 GB at
  27 theories, nearly all of it recomputing held-out scores and pool predictions.
- **Partitions and fairshare.** Heavy use of `normal` lowers the group's fairshare, so its short
  queues won't last (PI). **Fits run on `mcfrank,owners`** (the lab node first: same CPU, never
  preempted; then idle owners nodes, preempted fits requeued, 4% of fit time lost in the review's
  sample), with `-C CPU_GEN:SIE`. **Steps and collection jobs on `mcfrank`.** `normal` only as a
  deliberate overflow, never in the default list (Slurm takes whichever listed partition starts a job
  first). How `owners` use counts toward fairshare is to be checked (`sshare` before and after a test
  day). A5 and B1 below also cut what is charged, wherever it runs.
- **Array sizing:** one memory size per array, 18 GB, and 36 GB for out-of-memory retries (peaks
  11.9 GB median, 14.2 GB largest). `DEFAULT_FIT_BYTES` becomes a hard limit inside a job, so it
  rises to match.
- **The step design's own failure modes** (review §5) are requirements: arrays submitted held and
  released after their continuation; steps `--no-requeue`, one at a time per chain
  (`--dependency=...,singleton` on a per-chain name); a guard's refusal writes STOP and exits nonzero;
  a theory's own failure exits 0 from its fit task; no mail per array task; and a dedicated test that
  two continuations at the design/collection boundary can't publish a second study.
- **Equivalence is tested, not asserted** (review §6): the same agent outputs and cached fits, run
  straight through and run with interruptions, must give the same ledger, history and export. The
  first such test is built (below); step mode adds one with a kill between each pair of
  state-changing actions.

**Optimizations, by what they need** (review §9-10):
- *No result changes (infrastructure amendment):* **A1** saved scores (built); **A5** compile in a
  short process, then sample at about 5.5 GB instead of 15 (identical posterior; prototype by the
  Sherlock session); A2 no per-trial log-likelihood in fits.
- *Draws change, scores shouldn't (a registered amendment, after tests):* **B1** folds reuse the full
  fit's adaptation, 3-9 times faster folds, about half of all fit compute; it waits for a noise
  baseline (the same fold, another seed) and a test on the leading theories. B3 fewer fold draws
  (250 a chain). Whether chain 1, fitted without B1, is analysed with stage 2's chains is part of that
  decision.

**Order:** A1 and its equivalence test (driver, done); the Sherlock session's B1 tests and A5
prototype; then the fit arrays and the coarse step mode with the safeguards above (driver); a
rehearsal from small to large; then one amendment (infrastructure, plus B1 if it passes).

**A1 as built** (`src/rsa_open/loop/scores.py`; the loop in `orchestrator.py`):
- Each theory's out-of-fold scores (or its failed fold's reason) and pool predictions are saved as
  computed, in `<loop>/.scores/`, keyed by everything they depend on (the theory file, the corpus,
  the fold assignment, the full fit's and the folds' settings, the folds' time limit; for
  predictions, the pool). A resume reads them back; the chance theory's scores too.
- A live theory no longer holds its fit. It keeps a summary (converged, problems, timing) and loads
  its draws from the cache only when a step needs them: the critique's incumbent, or new out-of-fold
  scores. A resume reads summaries from the cache's sidecars (a cached refit of a near miss is the
  one used, as `loop_fit` decides) and loads no fit for a theory whose predictions are saved.
- Results are unchanged: both are deterministic given the cached fits, and the equivalence test
  checks the ledger, history and export of a crashed-and-resumed loop against an uninterrupted one,
  with the restore made to fail if it recomputes any score or loads any fit.

**The Sherlock session's last tests** (review §10, 2026-11-04): B1 (folds reuse the full fit's
adaptation) kept the winner, the order and the prunes on chain 1's leading theories, with
differences the size of a different seed's, and saved about 45% of fold compute on one partition
(9.7 to 4.4-5.7 fit-hours for 25 folds). Folds on the loaded lab node ran 2-3 times slower than on
`normal` (unexplained; perhaps memory bandwidth), which the arrays' throttle and `owners` overflow
absorb. A5 is not yet prototyped. B1 stays a PI decision for the amendment.

**Step mode as built** (2026-11-04; off unless `STEP_MODE=1`, so local runs are unchanged):
- *Fits* (`src/rsa_open/loop/fit_array.py`, `scripts/rsa/slurm/fit_array.sbatch`). With
  `RSA_OPEN_FIT_BACKEND=slurm` a fit that isn't done raises `FitPending`; the loop collects a
  phase's pending fits (`collect_pending`) and raises `FitsPending`. An array task runs the local
  backend's own child (time limit, peak memory) and writes `<fit>.mem<GB>.result.json`; the next
  step reads it as the local backend's outcome (ok, the theory's failure, timeout, killed: retried
  at 36 GB). A fit submitted twice with no result is an infrastructure failure. Tasks get
  `--export=NONE` and three paths, never `.secrets`.
- *Submission* (`src/rsa_open/loop/slurm.py`). One held array per memory size on `mcfrank,owners`,
  `-C CPU_GEN:SIE`, throttled at 24; then the continuation (`afterany` on every array the step
  submitted, early ones included, and `singleton` on the chain's job name); then release; one line
  in `<cell>/steps.jsonl`. It refuses, writing STOP (and so mail), when the cell is stopped, after
  80 steps, or with over 1,500 jobs queued; a failed call cancels what was submitted and STOPs.
- *Rounds as phases* (`orchestrator.py`). In step mode a round keeps a journal,
  `round_<k>/journal.json`: the round's starting standing and critique, every agent session, name,
  admission and ledger rejection, each written the moment it is known. Each step replays the round
  from the journal (no agent runs twice, no ledger line is written twice) up to the next phase whose
  fits aren't done: the first candidates (submitted while the retry agents work), the retries, the
  admitted theories' folds (submitted before the repair agents start), the repairs, and the scoring.
  A resume continues a journaled round instead of setting it aside. Setup puts the chance theory's
  fit in the seeds' array. The design's fits (`chain.design_context`) are one array too.
- *The step job* (`open_live.sbatch`, `open_rehearsal.sbatch`, `_step_mode.sh`). Each step writes
  `<cell>/continuation.sh` (submits the same script as from a login shell, `--no-requeue`, on
  `mcfrank`, 8 cores, 32 GB, 2 days) and `<cell>/step_settings.sh` (the chain's settings, sourced
  by the next step: `sbatch --export` cannot carry `mcfrank,owners`). `run_chain` exits 6 when it
  handed over (the job exits 0), 7 when it refused (STOP written, mail), and 5 when paused for time
  (the step submits its own continuation). A step that dies leaves no continuation: Slurm's FAIL or
  TIME_LIMIT mail, and `python -m src.rsa_open.status` shows the chain idle; relaunching resumes it.
  `open_launch.sh` takes `STEP_MODE=1` (one job per chain, no ladder).
- *Tests.* Fast: `test_rsa_open_fit_array.py`, `test_rsa_open_step_submit.py`,
  `test_rsa_open_step_journal.py`, `test_rsa_open_step_chain.py` (exit codes, the mode's guards,
  and a live chain re-run from its start by every step deploying and publishing once),
  `test_rsa_open_step_scripts.py` (the continuation against a stand-in sbatch). Slow, on Sherlock:
  `test_rsa_open_step_equivalence.py`, a two-round loop run as steps (arrays run locally between
  them, a fresh loop object each step, the local backend made to fail if called) against the same
  loop run straight through: the same ledger, history and export, and each agent called once.

**After SRCC's reply of 2026-11-05 and the rehearsal (review V3 §11), as built on 11-05.** Mark
Yoder (ticket #106983) wasn't sure how the agents policy reads for this, said it would be settled by
trial and error (with the exact message of any rejection), called the PI's plan "about right", and
asked to avoid `sbatch --wait`, which holds resources idle. The plan: a person starts one job per
experiment; the job runs the agents and sends out arrays of fits only; it then starts the next phase's
job, with a low, discrete cap on stages. Step mode already never waits; what changed:
- *One experiment per launch.* A step a person submits (no `STEP_SETTINGS`) records `launch.json`:
  the experiment it runs and the steps before it. When that experiment's loop exports, the chain
  writes STOP saying how to launch the next and exits 8 (the job exits 15, so Slurm mails). A launch
  refuses to start while another step of the chain is queued or running.
- *Step budget per launch:* 40 (the rehearsal took 2-3 a round, plus 3: about 18 for 5 rounds).
- *Folds after their full fits* (`fit_array.FoldAfter`): a starting theory's folds, and a repair's,
  are submitted with its full fit as an array that starts when the full fits end, so no step only
  submits folds (rule 10's 10-minute minimum; Mark's 20). A task without its full fit, or a repair
  whose fit did not converge, fits nothing; a failed fold is left to the loop. Results can't change.
- *Memory tiers 18, 36 and 72 GB*, a theory fitted before starting at the tier that holds its
  recorded peak with margin (chain 1's experiment-3 folds reached 28-32 GB).
- *Rehearsal findings:* a step never fits in its own job, an unlimited fit included (it goes to an
  array with no limit); a session killed with its step is set aside (`.killed_<dir>_<i>`) and run
  again from its brief alone, its start journaled first, the critique's session too; a loop before
  its first scored step is emptied, not set aside, each step; the equivalence test sets a fit limit.

## Appendix: v2 and v1, kept for the record (superseded where §0a-§0b say otherwise)

## 0. What changed in v2, and what waits for SRCC

The PI has asked SRCC (2026-11-03) about the three questions in the review's draft. Until the
answer, v2 builds only what no answer can rule out.

| | v1 | v2 | waits for SRCC? |
|---|---|---|---|
| Restarting a failed driver | automatic, up to a cap | **a person (or an attended session) resumes**; failure mail and a status command make it quick | yes: automatic resume only if SRCC is content (Sherlock's rules on unattended agent sessions) |
| The fit backend | a pool of workers reading task files | **native Slurm: one job per fit** (`sbatch --wait`), if SRCC allows a job to submit jobs; the file queue only as the fallback, with the review's four fixes | yes: which backend |
| CPU types | accept last-digit differences | **fits run only on the lab node's CPU type** (`-C CPU_GEN:SIE`, about 250 nodes), every fit's node and CPU recorded; time limits decide admission, so hardware must not | no |
| Where drivers run | 7 on the lab node, 3 CPUs, 24 GB | **`normal` jobs (never preempted, up to 7 days), about 8 CPUs and 48-64 GB**, sized from chain 1's rounds 4-5; a few on the lab node | no |
| The design's EIG search | inside the driver | **its own 24-core job** (with the fit backend's answer) | yes, as a job submitted by a job |
| Fit jobs' environment | not addressed | **a clean environment**: no `.secrets` keys, only what a fit needs (a fit runs agent-written code; Slurm passes the submitter's environment by default) | no |
| A round, a chain | 3.5 h; 3 days | **about 5-6 h** (every round so far had a retry or a repair, which is sequential); a loop 27-32 h; a chain about 4 days; stage 2 about 5-7 days with the reviews. The gain is chains at once, not shorter rounds | no |

**Built first, whatever SRCC says (done 2026-11-03, on `auto-psych-open`, not on chain 1's tag):**
- *The status command:* `python -m src.rsa_open.status --work-root <root> [--squeue] [--json]`:
  every chain's state, current stage, experiments (design, study, data, loop rounds and leader),
  STOP reasons, studies recruiting, agent and participant spend; a chain whose files haven't
  changed for 2 h (`--quiet-hours`) shows QUIET. Files only; `--squeue` adds one `squeue` call.
- *Failure mail:* `open_live.sbatch` has `--mail-type=FAIL,TIME_LIMIT`, and every STOP now exits
  nonzero so that Slurm mails it (10 finished, 13 awaiting review, 14 test deploy up, a failure its
  own code); a pause for the ladder's next job, and a later job that finds the STOP, exit 0.
  `MAIL_USER` and `MAIL_TYPE` (e.g. `NONE`) pass through the launcher. In `sacct` those three
  stops show as FAILED: they aren't failures of the job.
- *Where each fit ran:* every fit's sidecar records `machine`: host, CPU model, Slurm job and
  partition (`fitting.machine`).
- *The backend seam:* a time-limited fit's run is one `FitBackend.attempt` (`fitting.py`);
  `RSA_OPEN_FIT_BACKEND` chooses it, `local` (the default) is today's child process under the
  memory scheduler, and `slurm` or `queue` raise "planned, not built" rather than fall back. The
  out-of-memory retry with twice the room now works through any backend.

Then the backend SRCC allows, a two-chain rehearsal, and the amendment.

The sections below are v1's text where the table doesn't override it.

## 1. What costs the time, measured on open chain 1

- **Failures and the wait for a person.** Experiment 2 lost about 37 h: two loops killed out of
  memory (one restarted from scratch), and 5 h before anyone noticed the first. Any failed job
  writes STOP and halts the ladder until a person looks.
- **Fits.** A round is about 6 candidate fits (60-80 min each) and about 30 cross-validation fold
  fits (40-60 min each), each on one core with 7-10.5 GB at its peak. They are independent of each
  other and are the bulk of a round's compute. Agents use all of their 60 minutes; the loop's own
  process holds about 16.6 GB with 15 live theories.
- **Serial chains.** One chain uses a 24-CPU, 184-GB node, and memory limits it, not CPU. The lab
  has one such node.
- **Not the constraint:** recruitment (studies filled in 79 and 120 min; up to 8 at once).

## 2. The proposal in one picture

```
 mcfrank (lab node, never preempted)              owners / normal (any node with room)
 ┌──────────────────────────────────────┐         ┌────────────────────────────────┐
 │ driver: open chain 2  ─┐              │         │ worker 1: claim → fit → result │
 │ driver: open chain 3   │ agents run    │  task   │ worker 2: claim → fit → result │
 │ ...                    │ beside their  │ files   │ ...                            │
 │ driver: fixed chain 4 ─┘ driver        │ ──────▶ │ worker N (1 core, ~14 GB each) │
 │   (design, collect, loop, scoring)    │ ◀────── │                                │
 │   ~3 CPUs, ~24 GB each                │  cache  └────────────────────────────────┘
 └──────────────────────────────────────┘   (.nc files on $SCRATCH, shared)
```

Two changes, one per bottleneck:
1. **Chains run at the same time**, each as a light driver job on the lab node: design,
   collection, the inner loop's logic, scoring, and its agents. No fitting is done there.
2. **Fits run on a pool of worker jobs** on `owners` and `normal`. A driver needing a fit writes a
   task file; a worker claims it, fits it and writes the result into the fit cache the driver
   already reads.

And a third, which helps on its own and with chain 1's successors whatever else is decided:
3. **Unattended recovery**: an infrastructure failure resumes by itself, up to a cap; a person is
   needed only for decisions (§5).

## 3. The fit pool

**Why a pool, not one Slurm job per fit.** Stage 2 needs a few thousand fits. One job each would
wait in the queue each time, and could hit Sherlock's per-user job limits. A worker pays the
queue wait once and then fits one task after another. Small jobs (1 core, about 14 GB) also
backfill on `owners` and `normal` far more easily than a 184-GB whole-node job. And the workers
are submitted once, by the PI from the login node, so no job submits jobs (to be checked against
`/etc/agents/AGENTS.md`, §8).

**Where it plugs in.** Every time-limited fit (candidates, folds, starting, carried and design
fits) already goes through one function, `_fit_in_child` in `src/rsa_open/loop/fitting.py`. It
receives a content-addressed cache path (theory file, corpus file and sampler settings, plus the
starting point), waits for memory (`FitScheduler`), runs the fit in a child process under its
time limit, and reports `ok`, `model` (the theory's own failure) or an infrastructure failure. The
pool replaces only the "run the child here" step: with `FIT_BACKEND=pool` the driver writes the
task and waits for its result; without it, nothing changes (the default stays local). Cache keys,
settings, seeds, time limits, the convergence gate, near-miss refits, failure classification and
every score stay exactly as they are.

**The protocol** (files on `$SCRATCH`, as the cache already is; renames are atomic there):
- *Task:* `queue/tasks/<fingerprint>.json`, written under a temporary name and renamed into place:
  the theory file, corpus file, settings, starting point, time limit, cache paths and the memory
  it needs (the theory's recorded peak × 1.25, as the scheduler estimates now). Its name is the
  fit's cache key, so a task requested twice is fitted once.
- *Claim:* a worker renames the task to `queue/claimed/<fingerprint>.<worker>.json`. Only one
  rename succeeds, so only one worker gets it. A worker takes only tasks that fit its memory.
- *Run:* the worker runs the same child it would locally (`_run_child`: one core, the fit's own
  time limit, its peak memory read as `VmHWM`), touching its claim every minute as a heartbeat.
  Time waiting in the queue never counts against a fit's time limit, as now.
- *Result:* the worker writes the `.nc` and sidecar into the cache (atomically, as now), then
  `queue/results/<fingerprint>.json` with the child's report and its peak. The driver, polling,
  returns exactly what `_fit_in_child` returns today.
- *Lost workers:* a claim whose heartbeat is more than 10 minutes old (a worker preempted on
  `owners`, killed, or its node lost) goes back to `tasks/` with its attempt count. Two
  infrastructure retries, then the driver raises as today. A fit killed out of memory goes back
  with twice the room, as `FitScheduler` does now, and is taken only by a worker large enough.
- *Size classes:* most workers get about 14 GB; a few get 32 GB for outliers and retries.

**The workers** are a Slurm job array of a generic `fit_worker.sbatch`: 1 core, `--mem` 14G (or
32G), `--partition=owners,normal`, a time limit of 1-2 days, exiting after an idle hour. The
launcher submits them in a ladder (each array after the last ends), as it submits chains now, so
the pool lasts for stage 2 without any job resubmitting itself. 48-64 workers is a first guess:
7 unsynchronized chains need up to about 36 fits a round each.

**On different hardware.** `owners` nodes vary in CPU. A fit computed on one CPU type can differ
in the last floating-point digits from the same fit on another, and NUTS amplifies that into
different (statistically equivalent) draws. That doesn't change validity or any cache key, and
every fit's result is kept, so the campaign stays reproducible from its cache. If it matters, the
workers can be limited to one CPU generation with Sherlock's node features.

## 4. The drivers

- One driver job per chain on `mcfrank`: about 3 CPUs and 24 GB, so all 7 fit on the node beside
  their agents. It runs `run_chain` as today, with `FIT_BACKEND=pool`.
- **Agents stay with their driver at first.** They mostly wait on the Gemini API; their
  self-checks (`check_candidate`, its quick fit) are short. A node-wide cap on concurrent agent
  sessions (a lock, like the study slots) keeps 7 chains' theorists from crowding the node. Moving
  agents to the pool is a later step, if the node turns out to be the limit.
- **The design's EIG search** uses every CPU of its job. With 7 drivers sharing the node, it would
  run on the driver's 3 CPUs (slower), or later as a pool task on a larger worker. The design's
  fits already go through the pool.
- The live path doesn't change: one study per experiment, up to 8 at once campaign-wide, a person
  once per chain.

## 5. Unattended recovery

Today's rule (§4a of the pre-registration) already says infrastructure failures resume from the
last finished stage; what's missing is doing it without a person.
- **Automatic resume:** a driver that fails for an infrastructure reason (out of memory, a lost
  node, an API outage, a killed worker beyond its retries) exits with a code that tells the next
  job in its ladder to resume, not stop. A cap (say 3 such resumes per experiment) then writes
  STOP. The loop already resumes from its last scored round, so a failure costs at most a round.
- **STOP stays for decisions:** a designer failure (§4a), the PI's review, a test deploy, the cap.
- **Notice in minutes, not hours:** Slurm's failure mail on every driver job (`--mail-type=FAIL`)
  and a one-screen status command over the whole campaign (`python -m src.rsa_open.status
  --work-root ...`: each chain's stage, round and last event; the queue's depth and age; workers
  alive; studies recruiting; spend so far), for the PI or the Sherlock session to run.

## 6. What it would buy (estimates, not measurements)

- **A round** becomes about the agent session, the slowest candidate fit and the slowest fold
  fit: roughly 3.5 h instead of 5-7. A loop of 5 rounds plus its starting fits: about 20 h instead
  of 38. An experiment: about 25 h; a chain of 3 experiments: about 3 days.
- **Stage 2** becomes about as long as its longest chain plus the 3 open chains' reviews: perhaps
  4-5 days instead of 4 weeks, if the cluster has room for the workers.
- **Compute:** about 200 core-hours of fits per loop, so roughly 4,000-6,000 core-hours for stage
  2, spread over `owners` and `normal`.
- **The PI's time:** the reviews of the 3 open chains' first paradigms, the studies' monitoring
  summaries, and decisions when something stops, not keeping jobs alive.

## 7. Build, test and rehearse

1. **Unattended recovery** (§5): exit codes, the ladder's resume rule and cap, failure mail, the
   status command. Small; tested with stand-in jobs.
2. **The pool** (§3): the backend switch in `_fit_in_child`, the protocol, `fit_worker.sbatch`,
   the launcher's worker ladder. Tests: two workers racing for one task, a stale claim requeued,
   the retry cap, an out-of-memory retry with twice the room, `ok`/`model`/infrastructure reports
   passed through, and an end-to-end fit through two local worker processes giving the same cache
   file as the local path on the same machine.
3. **A simulated rehearsal on Sherlock:** two chains at once (one per arm), trimmed loops, a
   small pool on `owners`, with a worker cancelled mid-fit and a driver killed mid-round. It
   passes when both chains finish with no person acting, and the record shows the requeue and the
   resume.
4. **An amendment** before stage 2 (infrastructure only, like Addendum B), with the tag stage 2
   runs on.

Roughly 2-3 days of building, then the rehearsal, while experiment 3 runs; nothing waits on it
until the gate.

## 8. To check on Sherlock before building

- **`/etc/agents/AGENTS.md`:** that drivers running coding agents inside jobs is allowed as now,
  and that the pool needs nothing it forbids (no job submits jobs; workers are submitted by the PI).
- **Queue waits:** a few 1-core/14-GB and a few 24-core/184-GB test jobs on `owners` and `normal`,
  and how long each waits; and how often `owners` preempts over a day.
- **Gemini's limits:** 7 chains can mean 40 or more concurrent agent sessions; the account's
  rate limits (requests and tokens a minute) at that load. The harness waits out usage limits, but
  a run that waits often is slow.
- **The lab node at 7 drivers:** memory and CPU of 7 loop processes plus agents (about 16.6 GB a
  loop process measured; agents not yet).

## 9. Decisions for the PI

1. Build this for stage 2: recovery first, then the pool?
2. Agents beside their driver at first, moving them only if the lab node becomes the limit?
3. Accept that fits on different CPU types differ in their last digits (or limit the workers to
   one CPU generation, with a longer queue)?
4. A cap of 3 automatic resumes per experiment before a person is asked?

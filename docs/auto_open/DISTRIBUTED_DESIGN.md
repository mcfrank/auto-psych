# Design note: running stage 2 across the cluster (v2, 2026-11-03)

**Status: a proposal; nothing is built.** v1 answered the PI's question of 2026-11-02
(`RUN_CHAIN1_STATUS_2026-11-02.md`, "Parallelizing stage 2"). **v2 takes in the Sherlock session's
review (`DISTRIBUTED_DESIGN_REVIEW.md`)**; the changes are listed in §0. Open chain 1 is not
affected: it finishes on its tag.

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

**Built first, whatever SRCC says:** the status command, failure mail on every job, the fits'
node and CPU in their sidecars, and the backend seam in `_fit_in_child` with the local backend
unchanged. Then the backend SRCC allows, a two-chain rehearsal, and the amendment.

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

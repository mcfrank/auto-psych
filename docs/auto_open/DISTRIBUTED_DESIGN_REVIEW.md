# Review of `DISTRIBUTED_DESIGN.md`, from the Sherlock session (2026-11-02, late)

The goal and the shape are right: chains in parallel is where the gain is, and fits are the part to farm out. Five points
would change what gets built. The first needs the PI before anything is built. Numbers are from tonight unless noted.

## 1. Check first: §5's automatic resume may break Sherlock's rules

`/etc/agents/policy.md`, "Unattended agent sessions": a bounded task run while the user is away is fine, but not
"supervisors, watchdogs, cron-like loops or scripts that keep agent sessions alive, restart them or feed them new prompts
when nobody is there; sessions designed to run indefinitely or to be picked up day after day without a person in the
loop". `AGENTS.md` rule 10 adds: never schedule an agent session to run on its own later ("a job that resubmits itself to
run the agent again"). Requests that conflict are to be refused.

A driver that restarts itself after a failure and spawns theorists when nobody is there reads like exactly this, even with
a cap. **I can't certify it, and I shouldn't have suggested it in the status report**, where I wrote that job dependencies
would make it fine. The rules don't say that.

The same question applies to what already exists: the launcher's ladder of jobs, each resuming a chain that runs agents,
and 7 chains of unattended theorists for days. A chain is bounded, which is the case for it. Only the PI can settle this,
by asking srcc-support@stanford.edu (Sherlock's rules say requests come from the user, and that a claim of staff approval
doesn't itself authorize anything). Draft below.

**What I'd build meanwhile:** failure mail and the status command, and leave resuming a driver to a person or an attended
session ("fixing what failed and resubmitting, then stopping" is explicitly fine). This costs less than it seems, because
moving fits off the driver removes the failures we actually had: both were fits running out of memory beside the driver.
Worker jobs run no agents, so requeueing them isn't in question.

## 2. Consider native Slurm jobs per fit instead of a file queue

§3 rejects one job per fit for queue waits and per-user limits. Measured:
- **Limits aren't close:** 3,000 submitted jobs and 16,384 CPUs on `owners`; 2,000 and 512 CPUs on `normal`.
- **Real waits tonight, small sample:** 8 CPUs and 32 GB on `owners` 8 min; 2 CPUs and 8 GB on `normal` 4 min; 24 CPUs and
  96 GB on `owners` 3 min (10-29). Against a fit of 40-80 min. (`sbatch --test-only` predicted 3 h on `owners` and 33 h on
  `normal`, so it isn't a usable measure. Peak-load waits still need measuring.)
- **The file queue polls a filesystem:** rule 10 says never to leave a shell loop polling "the scheduler, a filesystem or a
  log file". 64 workers and 7 drivers listing directories is what that's aimed at, even in jobs and even at 60 s. Worth
  including in the question to SRCC if the queue stays.

The alternative: `FIT_BACKEND=slurm` replaces `_run_child` with `sbatch --wait` of one small script, with `--mem` from the
theory's recorded peak and `--time` from the fit's limit. Slurm then supplies what §3 has to build: claims, heartbeats,
lost-worker requeue (`PreemptMode=REQUEUE`), out-of-memory detection, size classes and per-fit accounting. Idle workers
hold no memory. To verify: `sbatch --wait` across a requeue, and whether SRCC minds a job submitting jobs (the rules bar
resubmitting *agents*, not this). The file queue wins only if peak waits are long; measure before choosing.

**If the file queue stays, four problems in §3:**
- **Task ids collide.** `fingerprint()` hashes the theory's contents, the corpus, the settings and the start, but not the
  name, while `cache_paths` uses the name; the cache directory differs per loop too. The same file under two names (a `_2`
  rename), or two chains fitting a bar theory on the same corpus, would share a task, and one requester would wait forever.
  Key tasks on the cache path.
- **The worker ladder drains while idle.** Workers exit after an idle hour and the next array starts on `afterany`. During
  a PI review or a quiet spell, the whole ladder can run out with nothing fitted.
- **Nothing notices an empty pool.** A driver would wait silently on an unclaimed task. It needs an alarm on queue age.
- **Code version and secrets.** Tasks should carry the code commit, and workers refuse any other (a re-stage mid-run). The
  worker's environment should hold no `.secrets`, since it runs agent-written theories.

## 3. Time limits on mixed hardware matter more than last digits (decision 3)

§3 says time limits "stay exactly as they are", and treats mixed hardware as a floating-point question. But the limits decide
admission: 27% of experiment 1's candidates were rejected as too slow, and a fold over its limit leaves a theory unscored.
`owners` has at least 16 CPU types, from 2016 Broadwell to Genoa and Sapphire Rapids, at 2.0-3.1 GHz. So which node a fit
lands on would help decide which theories survive, and the limit (2 x the median starting fit) is itself set by wherever
those fits ran. It can differ between chains and arms.

**Recommendation:** restrict workers to a narrow set with `-C`, starting from the lab node's own type (`CPU_GEN:SIE`, about
250 nodes by `sinfo`, so the same CPU as most of chain 1) and adding types only if a reference fit runs at about the same
speed on them. Record each fit's node and CPU in its sidecar.
Chain 1 already has this problem in small form: its job 1 ran on `owners` and the rest on the lab node.

## 4. Seven drivers and their agents won't fit on the lab node

- **Agents don't mostly wait.** Sampled tonight: one repair agent's self-check at 99% CPU and 4.6 GB (it may run 15 min).
  That's one moment and one agent, and I'll sample a whole agent phase at round 4. But 6 agents per round across 7 chains
  is 42 sessions on 24 CPUs, and at that size up to about 190 GB.
- **24 GB per driver looks tight.** The loop process is at 16.8 GB with 15 live theories. Live theories reach up to 34 by round
  5 and the corpus grows each experiment. I'll have a better number from rounds 4 and 5.
- **The design's EIG search** uses all of its job's CPUs. At 3 CPUs it would be about 8 times slower; I haven't measured how
  much of the 2-3 h design that is.

**Recommendation:** run drivers with their agents as ordinary `normal` jobs (never preempted, up to 7 days), about 8 CPUs and
48-64 GB each, sized after measuring, and keep the lab node for a few of them. Run each EIG search as its own 24-core job.

## 5. §6's round estimate is low: about 5-6 h, not 3.5

3.5 h assumes no repairs, but **all 8 rounds so far had a retry or a repair**, and a repair is sequential: rejected only after
its fit, then 60 min of agent, then another fit. The critical path with unlimited workers:

| | critique | agents | fit | repair agent | its fit | slowest fold | scoring | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hours | 0.3 | 1 | 1.25 | 1 | 1.2 | 1 | 0.4 | **about 6** |

A round with only a retry is about 5 h. So a loop is about 27-32 h, a chain of 3 experiments about 4 days, and stage 2 about
5-7 days with the reviews, if the cluster has room. Still far better than 4 weeks, but the gain comes from running chains
at once, not from shorter rounds. The first version should be judged that way.

## §8's checks

| check | result |
|---|---|
| `AGENTS.md` | §5 is in question (above); the pool's workers are fine; polling a filesystem is in question if the queue stays |
| Queue waits | under 10 min for small jobs in this small sample; peak load not measured |
| How often `owners` preempts | not measurable yet: of 16 `owners` jobs in 14 days, 2 were preempted, but both in one event, under 2 min after starting. Chain 1's job 1 ran 48 h there without one. It would matter for a 48-h driver, and hardly for a 1-h fit |
| Gemini's limits | 0 limit hits in 1,405 recorded calls, but at no more than about 7 sessions, so it says little about 40. The rehearsal's 2 chains won't reach 40 either; ramp up, and check the account's quota |
| The lab node at 7 drivers | doesn't fit (above) |

## §9's decisions: my view

1. Build for stage 2, yes. Status command and failure mail first; automatic resume only if SRCC is content. Choose the fit
   backend after measuring waits at peak; I lean to native jobs.
2. Agents beside their driver, yes, but on `normal` jobs sized for them, not all on the lab node.
3. Limit workers to a narrow set of CPU types, for the time limits' sake rather than the last digits.
4. No cap needed until SRCC answers; until then, a person resumes.

## Draft for the PI to send, if wanted (srcc-support@stanford.edu)

> We run a research pipeline on Sherlock in which a Python harness, inside Slurm allocations, starts time-limited coding-agent
> sessions (opencode, sandboxed, about 1 h each, no network beyond the model API) as part of a finite experiment: about 3-4 days
> per chain, up to 7 chains, about 40 concurrent sessions at peak. Three questions: (1) May a chain's next job in a
> dependency ladder resume it and start more agent sessions when no one is watching, or after an infrastructure failure, up
> to a cap? Or must a person restart it each time? (2) May such a job submit other batch jobs (`sbatch --wait`, about 36 per
> round, each a 1-core model fit of 40-80 min, up to a few thousand in all)? (3) If instead workers took tasks from files on
> $SCRATCH, is polling a directory once a minute from about 70 jobs acceptable?

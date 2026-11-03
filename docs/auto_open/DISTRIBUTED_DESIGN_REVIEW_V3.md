# Review of `DISTRIBUTED_DESIGN.md` v3 (2026-11-03)

Written by Claude Opus 5.5 (the Sherlock session) from chain 1's round 4-5 logs, the code, Sherlock's rules and `sbatch
--test-only`; not reviewed by a person. I've read only v3's summary of SRCC's reply, not the reply itself.

**Bottom line: arrays of fits and no orchestrator are the right direction, but don't start building step mode yet.** Two
things need settling first: a written Sherlock rule that covers follow-up jobs starting agents (§1), and the cost of
restarting the loop at every step, which measurement puts at about 10 times v3's figure (§2). I'd use 3-4 coarse steps a
round with the agents inside a step (§3). About a day of checks would settle most of the open numbers (§7).

## 1. Rule 10 still reads against the agent part

`/etc/agents/AGENTS.md`, numbered rule 10: "Never schedule an agent session to run on its own later (`scrontab`, `--begin`,
a job that resubmits itself to run the agent again)". A step that submits its own continuation and agent arrays matches
that wording. `policy.md` also bars scripts that "restart them or feed them new prompts when nobody is there".

The conditions of use say numbered rules are requirements, and that a user's report of staff approval doesn't authorize a
departure: the tool stops the conflicting part and asks the user to check with support. So this isn't something I can settle
by reading Mark's reply second-hand. The review's draft did tell SRCC about agent sessions, so he may well have meant this,
but v3's summary speaks of "a job that submits other jobs", not of one that starts agents.

**To ask, in the ticket or at office hours (Tuesdays 10:00, Thursdays 15:00):**

> Rule 10 of /etc/agents/AGENTS.md says never to use "a job that resubmits itself to run the agent again". Our design has
> each step job submit a follow-up job, with no person present, and those follow-up jobs start time-limited sandboxed agent
> sessions, as part of a finite experiment of about 4 days a chain, with step budgets and a STOP file. Does your answer
> cover this? If so, could the rule say so, or could you tell us the wording to record, since coding agents on Sherlock are
> told to refuse it otherwise?

Until then, fit arrays and steps that start no agents are fine to build and test. Agents starting from a continuation are
not something I'd run. Rule 10 also says work inside a job should last at least 10 minutes, ideally 30, so "small, minutes"
steps conflict with it at roughly 1,000 steps for stage 2; §3 fixes that too.

## 2. Each step is a resume, and a resume costs far more than v3 assumes

v3: "a few minutes of scheduling between steps... about 30 minutes a round". Each continuation is "the same code", so it
goes through today's resume path, which rebuilds every live theory (`resume`, `orchestrator.py:810-815`).

| measured | value |
|---|---|
| round 4, the resumed one: the "resumed" line to the critique's first file | **about 35 min, one core** (01:10-01:45), and the load before that line isn't separated from re-staging |
| round 5, not resumed: scored to the critique's first file | **under 1 min** (08:29-08:29), so this cost comes from the resume |
| the loop's memory after a resume, 21 theories | **19.3 GB**, about 0.9 GB per theory; about 31 GB at 34 by extrapolation |

At 6-8 steps a round that's **about 3.5-6 h a round on top of 5-6 h, not 30 min**, and each step holds 20-30 GB, so it isn't
small. I haven't profiled which part dominates: the `.nc` loads, `pool_predictions`, or re-reading fold results.

**So step mode needs more than checkpoints.** It needs a `Live` that doesn't hold fits or recompute predictions: pool
predictions and fold scores saved to disk, and fits loaded only where needed (the critique's incumbent, say). That's in the
registered `orchestrator.py`, and I'd expect it to take the "3-4 days" past what v3 allows. The estimate needs redoing once
it's profiled.

## 3. Use fewer, longer steps, with agents inside them

Agent arrays are the part of v3 I'd drop. Round 4 measured six theorist sessions at 3.7-4.7 GB and 3-7% of a core in most
samples, with 1-3 self-checks at a time. Six 60-minute array tasks at about a core each are the under-use Mark warned about,
and each needs a done-marker, a restored directory and a preemption rule. `_spawn_all` already works.

A round could be:

1. **Agent step** (lab node, about 8 CPUs, 32 GB, about 1.5-2 h): critique, then theorists. It submits the first candidates'
   fit array as soon as they're written, runs any retries itself, and hands off to a continuation that waits on the array.
2. **Admission step** (about 1 h if there are repairs): admits, submits admitted theories' fold fits at once, and runs repair
   agents itself while the folds run. It submits the repairs' fits.
3. **Second admission and scoring step(s)**: admit the repairs, fit their folds, score, prune.

That's 3-4 steps a round instead of 6-8. Each lasts long enough for rule 10, nothing waits idle, and most of the overlap
measured now is kept. It also needs fewer state checkpoints, and no agent state beyond the round's phase. All of it still
depends on §1.

## 4. Sizing the arrays

- **One `--mem` per array.** "`--mem` from the theory's peak" means an array per size, or size classes. I'd use **18 GB for
  everything and 36 GB for out-of-memory retries**. Peaks across 31 fits: 11.9 GB median, 13.1 GB 90th percentile, 14.2 GB
  largest. v1's "about 14 GB" would kill the largest.
- **`DEFAULT_FIT_BYTES` (10 GiB) would now be a hard limit.** Today it's only a reservation, and a fit may exceed it. Under a
  job's memory limit a new theory's first fit, at a median peak of 11.9 GB, would be killed at 10 GiB, possibly late in a
  60-80 minute fit (I haven't measured when the peak comes). New theories are every candidate, so this would be routine.
- **Drop `normal` for fits.** It has 8 GB per core, so an 18-GB fit is charged 3 CPUs. `-C CPU_GEN:SIE` leaves it 35 nodes
  (249 on `owners`). `--test-only` today: **about 32 h** on `normal`, 3.3 h on `owners`; those estimates can run high (one
  predicted 3 h for a job that waited 3 min). Its limit of 512 CPUs a user would nearly be reached at `%24` x 7 chains x 3.
- **Consider `--partition=mcfrank,owners`.** The lab node is the same CPU type, and isn't preempted.
- **All fits are then preemptible and uncheckpointed.** A preemption costs up to 80 min of that fit. I still have only one
  preemption event on record, so the rate is a real unknown (§7).
- **`normal`'s limit is 48 h, not 7 days.** The partition allows 7 days, but the QoS caps at 2 days, as `CLAUDE.md` says. The
  status report, my review and v2 all had it wrong. It now matters only for collection jobs, which fit.

## 5. Failure modes a step design adds

- **A crash between submitting the array and the continuation** leaves a chain with no job and no later mail. Submit the array
  with `--hold`, then the continuation, then release it, or make the step fail loudly on any `sbatch` error.
- **A requeued step can submit everything twice**, then two continuations race over the ledger. Steps should be `--no-requeue`
  on the lab node. `--dependency=afterany:<arrays>,singleton` with a per-chain job name makes Slurm itself allow one step per
  chain at a time, a simpler guard than a job count.
- **A duplicate must not publish a second study.** This is the real-money case; give it its own test: two continuations at the
  design/collection boundary, and a kill after publishing. I haven't checked how `live.py`'s guards behave across jobs.
- **A guard's refusal** (the queued-job limit, the step budget) must write STOP and exit nonzero, so that it mails.
- **Mail and exit codes:** a theory's own failure should exit 0 from its fit task, and arrays shouldn't mail per task, or
  routine rejections will bury a real failure.

## 6. Test the equivalence, and the claim that this buys concurrency, not speed

- **v3's rehearsal has no pass criterion.** "Unchanged" should be tested: feed the same recorded agent outputs and cached fits
  through local and step code, and require identical ledger, history, standings and export. Add a kill between each pair of
  state-changing actions. Addendum-style "infrastructure only" depends on this.
- **Time limits now run on other people's nodes.** Admission depends on them, so compare one reference fit on the lab node and
  on `owners` SIE nodes under load; the limit is derived from starting fits within a chain, which helps but doesn't settle it.
- **A chain stays about 4 days, as now.** All the gain is chains at once, and v3 says so. A round can't go much below its
  agents plus its slowest fits: round 4 ran 8-9 fits at once and took 7.3 h against round 3's 6.7 h with 7, so a2 shows no
  visible gain on one comparison. One fit was alone at 07:11 and 07:26, and the step wasn't scored until 08:29. The critique
  also varies: 26 min in round 4, 58 min in round 5.
- **A fallback that needs no rewrite:** each chain in its own `owners` job on today's code, with resume and a ladder, at about 12
  CPUs (8-9 fits, plus self-checks) and 184 GB. That's what Mark advises against, and a preemption would cost up to a round
  plus the resume. I wouldn't plan on it, but it's worth pricing in case step mode isn't ready at the gate: it depends only on
  the queue wait for that size. It's the PI's call.

## 7. Before building (about a day, mostly waiting)

1. The rule 10 question (§1). PI.
2. Profile a resume on chain 1's data: `.nc` loads vs. predictions vs. fold results. Sherlock session; a small job.
3. One array of about 24 real reference fits at 18 GB, `-C CPU_GEN:SIE`, on `owners`: gives queue wait, preemptions and speed
   against the lab node in one test. Each lasts over 20 min, as Mark asks. Sherlock session, if the PI agrees; about 24 x 80 min
   of compute, no agents, no Prolific.
4. Re-estimate the build and the round's overhead from 2 and 3, then decide on step count.

## 8. The note itself

v3 keeps §§0-9 from v1 and v2, which still present what §0a supersedes: `sbatch --wait`, drivers on `normal`, the worker pool,
7-10.5 GB, "agents use all of their 60 minutes", 3.5 h rounds, and §9's decisions about a pool and resume caps. v3 has no
decision list of its own. I'd move the old text to an appendix and add one, so the amendment is written from a single current
design.

Decisions I'd put to the PI: agents inside steps or as arrays (I recommend inside); steps per round; `owners` plus the lab node
only for fits; whether the §6 fallback stays open.

## 9. Further optimizations (added at the PI's request)

**Where the compute goes** (experiment 2's loop through round 4, from its sidecars; 46 theories, 271 fits, 230 core-hours):

| | warmup | sampling | total | share |
|---|---:|---:|---:|---:|
| fold fits (225) | 124 h | 53 h | 177 h | 77% |
| full fits (46) | 37 h | 16 h | 53 h | 23% |

So fold warmup alone is 54% of all fitting, and warmup is 70%. A fold's 300 warmup iterations cost 2.3 times its 500 sampling
iterations. Experiment 1 looks the same (29 h full, 145 h folds). Median minutes: full fit 68, fold 47. Stage 2 at about 230 h
a loop is the driver's 4,000-6,000 core-hours. On `owners` those hours aren't what's scarce; time per round and queue are.

Most of the size estimates below are **estimates, not measurements**, and I'd test them first (§10).

### A. No result changes: an infrastructure amendment is enough

| | change | evidence | gain |
|---|---|---|---|
| A1 | Save CV scores and pool predictions to disk, so a restart needn't rebuild them | a resume costs about 35 min; `standing()` re-reads 5 fold fits per theory (job 50929501 will split this) | most of §2's 3.5-6 h a round |
| A2 | Stop computing and storing the per-trial log-likelihood, keeping only the zero-probability check | a fold computes it for 2,000 draws, then the score recomputes held-out values separately from 400; the loop loads fits without it. I found no reader in the loop, but this needs a full check, the oracle and gate included | `.nc` files from 57-145 MB to a few MB; less memory and time, size unknown |
| A3 | Find what makes up a fit's 12-14 GB (compilation, sampling, log-likelihood, the write) | a mid-run fit already sits at 10-13 GB, so A2 alone may not fix it | each GB off adds fits to a node |
| A4 | A persistent JAX compilation cache, if compile time turns out large | the reference fits' wall time less their sampler time will show it | unknown |

### B. Draws change, scores shouldn't: needs testing and a registered amendment

| | change | evidence | gain |
|---|---|---|---|
| B1 | **Folds reuse the full fit's adapted mass matrix and step size**, with a short warmup | fold warmup is 54% of fitting. A mass matrix affects efficiency, not the target posterior, and the convergence gate still applies. Needs the full fit to save its adaptation, which it doesn't now | **perhaps 30-45% of all fit compute**, shorter stragglers |
| B2 | **Parallel chains: 4 CPUs a fit, not 4 chains vectorized on one** | `chain_method="vectorized"` on one core today. Steps per iteration are the slowest chain's: 15.4 against a mean of 8.7 in one fit's warmup; in sampling 3.2-4.0 of 4 chains are active | **wall time per fit about 4 times shorter** (68 to about 20 min, 47 to about 15), so the largest gain in round time; core-hours same or lower |
| B3 | Fewer fold draws, say 250 a chain | folds draw 2,000 and score 400 (`MAX_HELDOUT_DRAWS`). The extra draws serve only the convergence gate. **Testable with no new fits** by thinning saved folds and recomputing standings | about 10% of fit compute |
| B4 | Warm-start carried theories from the previous experiment's fit | starting fits' median rose to 125 min. **Interlock:** the admission limit is 2 x the median starting fit, so warm starts would collapse it unless it's set another way | 2 h or more off each loop's start |

**Together (rough):** B1-B3 could halve fit compute, and with B2 a round might go from 6-7 h to about 3.5-4 h: critique, 60 min of
theorists, fits, then a repair's agents and fits. That would put a chain nearer 3 days than 4. Agents would then be the
largest part of a round.

**Two cautions:**
- B2 multiplies CPUs per task, so the array throttle and the rule about threads matching the allocation have to follow it; the
  XLA device count must be set before JAX starts.
- If chain 1 counts with stage 2's chains in any analysis, B changes make them differ in sampler settings. That's a question for the
  PI and driver before the amendment, even if scores agree.

### C. Considered and not recommended

- **Dropping theories after 1-2 folds:** only 19-25% of live theories are over 100 nats behind at the last step, and it changes selection.
- **Fewer rounds, fewer folds, a smaller E:** these change the method. Available only by revision.
- **A shorter theorist limit:** run C's 40-minute limit left 19 of 23 sessions with no file. A shorter critique limit may be safe (26 and 58 min
  so far), but it's the PI's call.
- **GPUs:** the `gpu` partition has a long queue, JAX is pinned for the CPU wheels, and these models' small arrays may gain little. Try it only if B
  falls short.
- **Single precision:** I found no x64 setting in the fit code, so fits should already run in float32.

## 10. Tests for §9 (small, no agents, no Prolific; not yet submitted)

1. **Thin saved folds** on a copy of chain 1's fits, recompute R-hat, ESS and standings at 250 and 125 draws a chain: B3, no fits.
2. **The reference full fit three ways:** vectorized (running now), `parallel` on 4 CPUs, `sequential` on 1: wall time, memory, and how far posterior means
   differ: B2.
3. **One theory's five folds** with and without the full fit's adaptation: sampler time, convergence, held-out scores: B1. Needs a small patch on a test copy only.
4. **Memory by stage** in one fit: A2, A3.

About 15-20 core-hours in all. Tests 1 and 3 need chain 1's fits copied, as the resume profile does.

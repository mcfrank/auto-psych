# Review of `DISTRIBUTED_DESIGN.md` v3 (2026-11-03)

Written by Claude Opus 5.5 (the Sherlock session) from chain 1's round 4-5 logs, the code, Sherlock's rules, `sbatch
--test-only` and the test jobs of 2026-11-03 (§10); not reviewed by a person. I've read only v3's summary of SRCC's reply, not
the reply itself.

**Update 2026-11-07 (§13).** The driver's pattern-weighted likelihood (786e609b) makes fits 30× faster on the committed literature but only 2.5-2.9× faster on chain 0's experiment 3 data, with no saving in memory (36-39 GB per fit): the agents' theories are now most of the cost. §13 has the measurements and where I think the rest of the time goes.

**Update 2026-11-07 (§11, §12).** The PI launched a real-agent segment in step mode (one experiment of two, 2 rounds of 3): it worked, with no session run twice and a clean stop after the experiment (§11). §12 estimates how long a full-size stepped chain would take: about 5-8 days, against about 10-11 for chain 0.

**Update 2026-11-04 (§11).** Step mode was built after this review. On replayed agents it reproduced chain 1's decisions; real agents remain blocked by rule 10.

**Update after the tests (§10).** Five results change the plan, and each corrects something I wrote below or earlier:

1. **Most of a fit's memory is compilation.** With compiled code loaded from the cache, a fit peaks at about 5.5 GB rather
   than about 15 GB, with an identical posterior. A compile step of its own could lift the memory limit (A5).
2. **Folds that reuse the full fit's adaptation took about 40-47% less time than the same folds at the chain's settings on the same
   partition**, on 5 theories with no convergence problems, which is about a third of all fitting. I first said "a third of
   the time or less" and "about half of all fitting", from comparing with the chain's own times; that overstated it (B1, §10).
   Scores moved no more than a new seed moves them, and the winner and order held, so the extra test B1 needed has passed. It
   still needs a registered amendment.
3. **Parallel chains gave little, not the 4 times I estimated** (B2). `normal` is usable, against my advice in §4.
4. **Compilation is also most of a B1 fold:** about 9 of 15 min and 10 of 13.5 GB, on the one theory where I could separate it.
   That makes A5 and A6 the next-largest saving.
5. **The chain's folds ran about 2-3 times slower than the same folds on `normal`,** at about the same number of steps. I don't
   know why (§10). If it's load on the lab node, running folds as separate jobs gains about that much with no change to the
   sampler.

Saving CV scores to disk would remove nearly all of a restart's cost, which is what made §2's steps expensive (A1).

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
small.

**Profiled since (job 50929501, 27 live theories, one core): 74 min, 24.9 GB.** Recomputing held-out scores takes 64.8 min (420
calls), pool predictions 8.9 min, loading fits 9.8 min. So at 27 theories it's about twice the real resume I measured at 21
(35 min). I can't fully explain that gap; the theory count alone doesn't account for it.

**So step mode needs more than checkpoints.** It needs a `Live` that doesn't hold fits or recompute predictions: pool
predictions and fold scores saved to disk, and fits loaded only where needed (the critique's incumbent, say). The profile
makes this look smaller than I'd feared: nearly all the time is in two results that can simply be saved. It's still in the
registered `orchestrator.py`, and the memory (24.9 GB) depends on not holding fits, which I haven't tested.

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
- **Keep `normal` for fits (corrected; I first said to drop it, from a `--test-only` estimate of about 32 h).** Real waits were
  under 4 min and 8-11 min for the reference fits, and no job there was preempted. The costs stand: an 18-GB fit is charged 3
  CPUs, its limit of 512 CPUs a user would nearly be reached at `%24` x 7 chains x 3, and `-C CPU_GEN:SIE` leaves it 35 nodes (249
  on `owners`). This is one day's sample. `--partition=normal,owners` is worth trying.
- **Consider `--partition=mcfrank,owners`.** The lab node is the same CPU type, and isn't preempted.
- **On `owners`, fits are preemptible and uncheckpointed.** Measured on one morning: 7 of 24 fits preempted, 6 of them within 8
  min of starting and all 6 on three neighbouring nodes (`sh04-17n24`, `n26`, `n27`), one after 42 min. About 71 min lost of about
  29 fit-hours (4%). All were requeued and finished. A non-requeuing job on `owners` lost everything (my thinning test).
- **Node speed varies.** Same fit, same steps: 69-73 min on most `owners` nodes, 71-74 on `normal`, 77 on the lab node with 8-9
  fits running, but 98 min on `sh04-13n25`. Since admission depends on a time limit, that 40% matters (§6).
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
- **Time limits now run on other people's nodes.** Admission depends on them. Measured: most nodes are 5-10% faster than the
  busy lab node, and one was about 40% slower than the rest (§4). The limit is 2 x the median starting fit, so this one
  straggler would have passed, but a slow candidate on a slow node might not.
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
2. ~~Profile a resume on chain 1's data.~~ Done (§2).
3. ~~One array of about 24 reference fits on `owners`.~~ Done (§4, §10).
4. Re-estimate the build and the round's overhead, then decide on step count. Driver.

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
| A1 | Save CV scores and pool predictions to disk, so a restart needn't rebuild them | **measured:** nearly all of a restart's 74 min at 27 theories is these two and the fit loads they need (§2) | nearly all of §2's 3.5-6 h a round |
| A2 | Stop computing and storing the per-trial log-likelihood, keeping only the zero-probability check | a fold computes it for 2,000 draws, then the score recomputes held-out values separately from 400; the loop loads fits without it. I found no reader in the loop, but this needs a full check, the oracle and gate included. **Measured:** 2.7 min of a 77-min fit | `.nc` files from 57-145 MB to a few MB; about 3.5% of a fit's time; little memory |
| A3 | Find what makes up a fit's 12-14 GB | **done:** compilation. Memory reaches about 12.7 GB within 8 min and stays there through sampling; with a cache hit the whole fit peaks at 4.9 GB (`VmHWM`; 5.5 GB by Slurm) | see A5 |
| A4 | ~~A persistent JAX compilation cache~~ | **withdrawn, my error:** `_env.sh` has had one since 2026-10-09 (the chain's is 16 GB). It doesn't help a first compile, and each fit's appears to be a first compile | none |
| A5 | **New: compile in a short process of its own, then sample in a fresh one that loads from the cache** | cached and compiled fits have **identical posteriors (largest difference 0.0 over 12 parameters)** and identical step counts, and the cached ones took 60 min against 70-74. Not built; numpyro compiles its warmup and sampling loops separately, so the way to compile without sampling needs a prototype | **sampling at about 5.5 GB instead of 15:** about 24 fits on a 24-CPU node rather than 8-9, and 1 CPU rather than 3 charged on `normal` for that part. Only for theories that compile like this one; the other theory tested peaks at 3.6-4.4 GB anyway |
| A6 | **New: one compile for a theory's folds**, if the data can be passed as arguments rather than compiled in | **measured on one theory:** rerunning 5 B1 folds with a cache hit took 5.4-7.3 min at 3.7-3.9 GB, against 14.9-15.7 min at 10.9-13.5 GB with a compile. So a compile is about 9 min and 10 GB of such a fold. One compile across folds isn't tested, and the folds' differing sizes need handling. The seed appears to be part of the compiled program: a new seed missed the cache | possibly half of a B1 fold again, and the memory limit with it |

### B. Draws change, scores shouldn't: needs testing and a registered amendment

| | change | evidence | gain |
|---|---|---|---|
| B1 | **Folds reuse the full fit's adapted mass matrix and step size**, with a short warmup | **measured (§10), against the same folds at chain settings on `normal`:** 22-25 to 13-17 min on three theories, 24 to 7-8 min on a fourth; 9.7 fit-hours to 5.7 over 25 folds of 5 theories (seed 1; seed 0's 4.4 includes a cache hit, about 5.2 without it), 65 of 65 with no convergence problems. Scores moved within seed noise; winner and order unchanged. A mass matrix affects efficiency, not the target posterior. Needs the full fit to save its adaptation, which it doesn't now | **about a third of fit compute** (about 40-47% of the folds, which are 77% of it); shorter stragglers |
| B2 | ~~Parallel chains: 4 CPUs a fit~~ **Sequential chains on 1 CPU instead** | **measured:** parallel took 51-52 min on 4 CPUs with only 1.7-1.8 busy on average, against 59-63 min for sequential on 1 CPU and 77 for today's vectorized. **My estimate of 4 times shorter was wrong.** The single-thread XLA flags may be limiting the parallel test; I haven't checked. Sequential saves warmup only (33-35 min against 50) | **sequential: about 20% of a full fit**, on one theory, different random numbers; less under B1, untested |
| B3 | Fewer fold draws, 250 a chain | folds draw 2,000 and score 400 (`MAX_HELDOUT_DRAWS`). **Measured by thinning one step's saved folds:** at 250, same winner and top 5, largest score change 0.69 (median 0.15). At 125, one theory fails the convergence gate, so 250 is the floor | about 10% of fit compute (half of fold sampling; estimated) |
| B4 | Warm-start carried theories from the previous experiment's fit | starting fits' median rose to 125 min. **Interlock:** the admission limit is 2 x the median starting fit, so warm starts would collapse it unless it's set another way | 2 h or more off each loop's start |

**Together (rough):** A5, B1 and B3 could about halve fit compute and lift the memory limit; A6 would take more if it works. I withdraw my figure of a round
at 3.5-4 h, which relied on B2. A round's length will still have a floor set by the critique, the 60-minute theorists and the
repair cycle, and I haven't re-estimated it. **My order: A1, A5 prototype, then B1, whose extra test has passed.**

**Two cautions:**
- ~~B1 isn't safe to adopt until it's tested on the leaders against a noise baseline.~~ **Done (§10).** My comparison here was also
  wrong: the 1.6 was over all blocks, while the 3.16 margin is on the newest block only, where the change was 0.36.
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

## 10. Test results (2026-11-03; no agents, no Prolific; all on copies of chain 1's fits)

All on `-C CPU_GEN:SIE`, the same CPU as the lab node. About 60 job attempts: 51 completed, 8 preempted (all on `owners`), 1 failed
(my script, below). One theory for the fit tests, two for folds, one loop step for thinning, one day of queues. Treat these as
strong leads, not settled rates.

**Queue and preemption (24 reference fits on `owners`, 18 GB; 4 on `normal`).** Waits: `owners` 4-11 min; `normal` under 4 min, and
8-11 min on the earlier failed attempt. Preemption and speed: §4. Fits with a compile peaked at 13.1-14.6 GB (`VmHWM`), about 15.5 GB
by Slurm's `MaxRSS`, so 18 GB holds; no convergence problems in any.

**Memory and compilation (A3, A5).** Six of seven requeued fits happened to find the cache written by earlier fits, and peaked at
4.9 GB. I predicted the result before testing it with two fresh fits: 4.92 and 4.93 GB (5.5 GB by Slurm), 60 min, posterior
identical to a compiled fit's. Two fits that started just as the cache filled (tasks 1 and 24, about 13-13.4 GB) got a partial hit;
that's my reading, not a measurement.

**A fit's stages (one theory, vectorized, 77 min):** 49.6 min warmup, 24.5 sampling, 2.7 log-likelihood, 0.2 writing. A 145-MB `.nc`.

**Chain method (B2), 2 fits each:**

| method | CPUs | minutes | warmup | sampling | largest mean difference from the lab fit |
|---|---:|---:|---:|---:|---:|
| vectorized (today) | 1 | 77 | 50 | 24 | 0.0 SD (same draws) |
| sequential | 1 | 59-63 | 33-35 | 23-25 | 0.014 SD |
| parallel | 4 (1.7-1.8 busy) | 51-52 | 26-30 | 20-22 | 0.014 SD |

**Fold reuse of adaptation (B1), all 5 folds of 2 theories, on `normal`.** "Score change" is the sum over the folds' held-out trials,
new less chain's, for about 58,000 trials:

| theory | warmup | fold minutes (chain's; see the correction below) | peak GB | R-hat at most | score change, all blocks |
|---|---:|---:|---:|---:|---:|
| `structural_alignment_pragmatic_tom` | 100 | 13.7-18.3 (45-50) | 10.9-13.6 | 1.006 | +0.16 |
| same | 50 | 13.7-18.8 (45-50) | 11.0-13.7 | 1.012 | +0.51 |
| `rsa_l2_singleton_feat_color_valence_l0` | 100 | 6.0-9.3 (64-73) | 3.4-3.6 | 1.007 | +1.60 |

No divergences; smallest bulk ESS 868. ~~Chain times are from the lab node under load, so the true ratio is about 10% smaller.~~
**Wrong:** the same-partition baseline below shows the ratio was far smaller than this table suggests.
Warmup 50 saves nothing over 100, so compile must take much of what's left. The second theory is slow in warmup, not on a slow node:
its full fit took 40,228 warmup steps on the chain and took about the same time here (124 min), so this is where B1 gains most.

**Thinning (B3), step 4's folds, 27 theories.** At 500 draws the code reproduces all 27 recorded scores exactly, which validates
the test. At 250: same winner and top 5, largest change 0.69, median 0.15, one swap at positions 15-16. At 125: same winner, but
`minimal_exemplar_diagnostic_mixture_l2` no longer passes convergence.

**Mistakes along the way:** my first reference array had no `__main__` guard and failed at once; my thinning job had `--no-requeue`
and was preempted, so I reran it on `normal`; my capture script crashed after finishing, on a field old fits lack, which cancelled
the fold tests by dependency, so I resubmitted them. None affects the numbers above.

**Noise baseline and B1 on the leaders (2026-11-03, later).** Round 4's top 4 theories and `rsa_l2_singleton_feat_color_valence_l0`,
all 5 folds each, on `normal`: the chain's own folds (seed 0), a refit at chain settings with seed 1, and B1 (reuse, warmup 100) with
seeds 0 and 1. The script applies the loop's `standing()` among these five. **Validation:** "chain seed 0" reproduces all 5 recorded
scores to 0.01, and reuse seed 0 reproduces the earlier run's changes (+0.21 and +0.36 on exp2).

Score on exp2, the block that selects, and the prune test (`diff > 2 × dse`):

| theory | chain, seed 0 | chain, seed 1 | reuse, seed 0 | reuse, seed 1 |
|---|---:|---:|---:|---:|
| `bipolar_boundary_spillover_topic_tom_2` | -12204.62 | -12204.79 | -12204.92 | -12204.82 |
| `polar_grounding_boundary_spill_tom` | -12207.79 | -12207.62 | -12207.73 | -12207.97 |
| `aspect_qud_boundary_tier_shadowing_tom` | -12210.19 | -12210.15 | -12210.23 | -12210.39 |
| `structural_alignment_pragmatic_tom` | -12212.03 | -12212.08 | -12211.82 | -12211.78 |
| `rsa_l2_singleton_feat_color_valence_l0` | -12336.35 | -12336.60 | -12335.99 | -12337.12 |
| second place: behind, 2 × dse | 3.16, 3.06 **pruned** | 2.83, 2.90 kept | 2.81, 3.14 kept | 3.15, 3.16 kept |

- **Winner and order: the same in all four.** Every other prune decision is the same too.
- **The one flip is seed noise, not B1:** second place was pruned only in the chain's own run, by 0.10 nats, and a new seed at the
  chain's settings keeps it. So that call was chance, as I flagged. Round 5 later pruned this theory anyway, by a wide margin.
- **Size of the noise:** on the four leaders, a new seed at chain settings moved an exp2 score by at most 0.17, and B1 by at most 0.30.
  The gap to the leader moved by at most 0.34 and 0.66. Averaged over seeds, B1 less chain is -0.17 to +0.26 by theory.
  `rsa_l2` is noisier under B1 (its two seeds differ by 1.13, against 0.25 at chain settings), but it's 131 nats behind.
- **The +1.6 I was worried about is resolved:** on E, a new seed at chain settings moved `rsa_l2` by +1.61, and B1 by +1.71 and +1.91.
  So the chain's own seed 0 was the unusual draw.
- **What this can't show:** with 5 theories and 2 seeds a method, a bias of a few tenths of a nat can't be ruled out, and `structural`
  moved up about 0.25 under both B1 seeds. It isn't a test of round 5's calls, whose theories aren't included. Those margins were 0.5
  (previous leader pruned), 1.0 and 1.6 (kept), the same order as this noise. Any 2-SE rule is arbitrary that close to its line.

Time, the same 25 folds on the same partition (mean minutes a fold; "steps" are warmup + sampling):

| theory | chain's record, lab node | chain settings, seed 1 | reuse, seed 0 | reuse, seed 1 |
|---|---:|---:|---:|---:|
| `aspect_qud_boundary_tier_shadowing_tom` | 44 (10,400 + 4,700) | 23 (11,000 + 4,500) | 13 | 15 |
| `bipolar_boundary_spillover_topic_tom_2` | 49 (10,400 + 5,200) | 25 (11,100 + 4,700) | 14 | 17 |
| `polar_grounding_boundary_spill_tom` | 47 (10,700 + 4,600) | 23 (10,800 + 5,100) | 13 | 14 |
| `structural_alignment_pragmatic_tom` | 47 (10,300 + 4,900) | 22 (11,200 + 4,500) | 6 (cache hit) | 15 |
| `rsa_l2_singleton_feat_color_valence_l0` | 68 (20,000 + 7,100) | 24 (21,700 + 7,300) | 7 | 8 |
| fit-hours, all 25 | 21.2 | 9.7 | 4.4 (about 5.2 without the cache hit) | 5.7 |

Convergence problems: 0 of 75. Peaks are unchanged by reuse, apart from the cache hit (3.8 GB).

**Unexplained: the chain's folds took about 2-3 times as long for about the same steps.** The lab node under load is the obvious
candidate: `rsa_l2`'s folds ran under the old code at up to 20 fits at once, and it shows the biggest gap, the others at 7-9. But a
full fit on the chain took only about 7% longer than on `normal` (77 min against 71-74), which doesn't fit that story, and these are
different times with different seeds. I haven't resolved this. It's worth a test, since memory bandwidth on a full node would change how
many fits to put on one node. The chain's fold records against how many fits were running at the time would be a cheap first look.

Cost: 91.7 core-hours charged, not the "about 36" I told the PI: each 24 GB task is allocated 4 CPUs, of which it uses 1. Waits were
long: `normal` was full at about 17:00-20:00 and the last folds ran about 5 h after the first, against minutes earlier today. So my
§10 queue figures, from one afternoon, are optimistic for an evening.

**Not tested:**

1. **A5's prototype:** compiling without sampling, then a fresh process's peak.
2. **A6:** one compile across a theory's folds.
3. **Why the chain's folds were slower** (above).
4. **B1 with B3 together,** and B1 on more than these 5 theories at more than one step.
5. **Whether the parallel test was limited by XLA's single-thread flags**, only if anyone wants B2 back.

## 11. Step-mode results (`HANDOFF_stepmode.md`, 2026-11-03/04, code at 9762aa8c)

Written by Claude Opus 5.5 (the Sherlock session); not reviewed by a person.

**Bottom line: step mode's Slurm side works and reproduces the live chain's decisions, but real agents have never run in it,
and rule 10 still blocks that.** On replayed agents it matched chain 1's experiment 2 through three rounds, and it recovered from
37 lost fit attempts, a cancelled fit and a killed step. SRCC has since said staff can't waive rule 10 (ticket #106983), so I
didn't run §2's rehearsal with real agents, which would mean jobs that submit their successor and start agents. The arrays, the
journal and the 18 GB/36 GB sizing would all carry over to the design in "What this means for the design" below, which
needs no self-resubmitting job.

### Tests (handoff §1)

- **Fast step tests:** 21 passed (job 51022228).
- **Slow step tests:** 12 passed, 1 failed in 1:36:30. The failure is `tests/test_rsa_open_step_equivalence.py`, and it's a test
  bug: line 92's `fit_time_limit_sec=None` sends every fit in-process (`fitting.py:397`), so the "stepped" loop never steps and
  `assert steps >= 4` fails.
- **Corrected copy** (that line set to 1800, outside the clone; job 51065065): **passed**. Same ledger, history and export; each
  session once; no fit inside a step. I didn't edit the test. It's the small synthetic loop.

### The stand-in rehearsal

`scripts/rsa/open_compute_checks/replay_step.{py,sbatch}`: the real `run_chain.main`, step exits, arrays, continuations and
journals, with each "agent session" copying what that session's agent wrote in chain 1's experiment 2. No model is called and
no agent starts. Steps ran on `owners,normal` with `STEP_TIME=12:00:00`, not the defaults' `mcfrank`, because chain 1 holds the
lab node. For the same reason every fit ran on `owners`, with `CPU_GEN:SIE`.

| run | what it does | outcome |
|---|---|---|
| smoke | 1 starting theory plus the chance model, 1 round of 2 | DONE, about 6.5 h, 5 steps |
| second chain | same as smoke, with a fit cancelled and a step killed | DONE, 8.5 h (about 1 h of that lost to the cancel test), 7 step jobs, no STOP |
| full | 4 starting theories, 3 rounds of 6 | DONE, 17.6 h, 10 step jobs, 133 fit tasks, no STOP |

**Equivalence with the live chain (full run):**

- After the starting fits and each of rounds 1-3: same leader, same live set, ELPD within 0.0008 nats on every block.
- Every line the replay added to the ledger (33, no duplicates) is in the chain's. The 6 chain lines it lacks for those rounds are
  later prunes of models that survived the replay's end, from the chain's two further rounds.
- The admission limit differed (7,052 s against 7,512 s), since it's measured from each run's own starting fits. No decision
  changed, but one could: it's the one input here that depends on hardware.
- The replay's last history step and its export are its own end-of-experiment prune after 3 rounds, so they aren't comparable
  with the chain's fifth-round end.

**Second chain against smoke:** ledger identical in order (67 lines), export identical, CV scores within 5e-7, and only
`rsa_l2_singleton_feat_color_valence_l0` differed. That's about float32 rounding, but I haven't established the cause.

**Failure tests:**

- **Cancelled fit** (`scancel` on one task): the next step exited 6 and resubmitted only that fit. No STOP.
- **Killed step** (`scancel` of a running step at 186 s): no STOP, and nothing was left queued, so a killed step ends the chain
  until someone relaunches it. On relaunch it finished with the same results, but `round_1/critique` ran twice (see finding 1).
- **Natural preemption:** all 37 lost attempts across the two chains (`owners`: 33 preempted, 1 node failure, 3 `REQUEUED`
  records) came back by themselves.

### What the handoff asked for

| item | result |
|---|---|
| Steps per round | 2 (round 3: 3, for a repair's fits), plus 2 for the starting fits and 1 to finish. Below the 3-4 I suggested in §3 |
| Arrays per step | 1-2 (an early array for first candidates or folds, then the rest) |
| Tasks per array | 1-30; 133 in the full run |
| Fit array start | About 5 min after submission, later for the last tasks of large arrays |
| Last fit to next step starting | Usually 4-12 min; once 31 and 69 min, both steps starting at 14:15 on 11-04 |
| Fit exits | All `COMPLETED` 0:0 apart from preemption and my cancel. None needed the 36 GB retry |
| Step exits | All 0:0 apart from my kill; no step was preempted (17 step jobs, 0 preempted, including 10 on `owners`) |
| `step_settings.sh`, `continuation.sh` | Written as designed; the commands they carried worked |
| `journal.json` | Holds the sessions, names and admissions; round 3 of the full run has 23 keys |
| Agent dir written twice | Never in the full run (26 sessions); once after the deliberate kill |
| Duplicate ledger lines | None |
| Step MaxRSS | 0.9-9.8 GB of 32 GB, **without agents** |
| Fit MaxRSS | Mean 6.7, max 12.5 GB of 18 GB in the full run |
| Time lost to preemption | 7.8 h of about 106 h of fit attempts in the full run (7%); median 11 min per lost attempt |
| Step durations | 37 s to 49.5 min in the full run (3 of 10 under rule 10's 10-minute line); all under it in the second chain. Replayed sessions take no time, so real steps would be longer. I didn't pad them |

**Is 32 GB right for steps, 18 GB for fits?** Without agents, 32 GB is plenty. For fits, 18 GB holds for everything in this
rehearsal, but **chain 1's experiment 3 had two folds of `bipolar_prototype_tier_contrast_tom` at 28 and 32 GB RSS (`ps`, 11-04
19:56)**, against 14.2 GB as the highest I'd measured before. Those would hit the 18 GB limit and retry at 36 GB, which leaves
little room. Experiment 3's job peaked at 176 of its 184 GB too. I'll look at this in the experiment 3 report.

**Does `owners` count toward fairshare? Not settled.** `RawUsage` rose 4,413,892 between about 05:45 and 23:28 on 11-04. In that
window billing-seconds were 4,449,410 on `mcfrank`, 2,454,316 on `owners` and 59,504 on `normal`. That fits `mcfrank` alone with
no decay, and it also fits all three with a decay half-life of about a week. I don't know the half-life, and reading the
scheduler's configuration falls under rule 9. `owners`, `owner` and `normal` all have `UsageFactor` 1.0. FairShare also moved from
0.298 to 0.900 in that time, which I can't explain. A clean test needs a stretch with nothing on `mcfrank`.

### Findings for the driver

1. **A kill mid-session repeats the session.** `_memo` (`orchestrator.py:240`) writes the journal only after `fn` returns, and
   `critique`'s `fn` is the session plus the PPC. With real agents that's a paid session again, and a candidate session could
   restart in a directory holding a killed session's partial files, which replay can't test. Consider saving the session's result
   before the PPC, and clearing the directory on restart.
2. **Fix the equivalence test's line 92,** and make `fit_cached` with `time_limit_sec=None` raise in step mode, so an in-process fit
   can't pass unnoticed.
3. **The design fit limit** (`cfg.design_fit_limit_sec`, `chain.py:292`): I didn't confirm it's always set. The same in-process
   path would apply.
4. **Steps before the first `history.json` set the loop aside each time:** `loop.abandoned_1` to `_3`, 149 MB each. Harmless, but
   each is another corpus copy.
5. **A killed or preempted step ends the chain without a STOP.** `--no-requeue` is right for agents, but nothing tells anyone.
   No step was preempted here, in 17 tries.

### Not tested (replay can't)

Real agent sessions across steps and how long they take; a step's memory with agents running; agent directories under real
agents; the design-stage array (`chain.design_context`); the collection boundary. Handoff §3's A5 prototype wasn't run either.

### What this means for the design

Rule 10 bars a job that resubmits itself to run the agent again, and scheduling an agent session to run on its own later. The
compliant version of this work looks to me like **one job per experiment or chain, started by a person, holding the driver and
agents, sending fits out as arrays and blocking with `sbatch --wait`**, with no job submitting a job that runs agents. It keeps
what this rehearsal showed works: an out-of-memory fit kills one task, and slow fits stop tying up 24 CPUs. The journal and
`ArrayBackend` should mostly carry over, waiting in-process instead of yielding, but I haven't built it; that's the driver's
call. Costs: the job's 8 CPUs and 32 GB sit idle during waits, a person launches each job, and a dead job waits for a person. The
PI wrote to SRCC again on 11-04; I haven't seen an answer.

### Handoff §4 items 1-2, and the sandbox (2026-11-05, code at b8775b4b)

Written by Claude Opus 5.5 (the Sherlock session); not reviewed by a person.

**Bottom line: the smoke replay's mechanics all worked; one slow test fails because of the test, not the code; and an agent in
the sandbox can't reach Slurm.** Since this section was first written, SRCC replied (Mark Yoder, 11-05: try some segments,
avoid `--wait`, send specific error messages). The PI decided to try the standard stepped version, which the PI will launch by
hand (handoff §4 item 3).

**Tests** (job 51213222, a clone at b8775b4b):

- **Fast** (`tests/test_rsa_open_step_*.py tests/test_rsa_open_fit_array.py`, not slow): 39 passed.
- **Slow** (`tests/test_rsa_open_step_equivalence.py`): failed, in the test's harness. `LocalSlurm.run_arrays` runs the array
  tasks in the test's own process, where the test has set `RSA_OPEN_FIT_BACKEND=slurm` and replaced `LocalBackend.attempt`. A
  `fold_after` task fits through `loop_fit`, so it picks the array backend and raises
  `FitPending: literal_listener: pending (18 GB)`. Full-fit tasks bypass the backend (`_run_child`), which is why the test
  passed before folds-after (b35f8090). On Sherlock, tasks run with `--export=NONE` (`slurm.py:193`), so they fit locally. The
  smoke replay's 20 `fold_after` tasks all returned `ok`.
- **Suggested fix:** run each task with the backend unset and the real `LocalBackend` (or in a subprocess with a clean
  environment), and keep the "never fit in a step" check for the steps themselves.

**Smoke replay** (handoff §4 item 2; `ROUNDS=1 CANDIDATES=2`, one starting theory; steps on `owners,normal`): mechanics only,
no timing analysis (PI, 11-05). All 28 jobs `COMPLETED` 0:0, so there are no failed job ids to report.

| check | result |
|---|---|
| A fold array held until its full fits end, then released and run | Yes: full fits 51213298 ended 10:55, folds 51213345 (`afterany` on them) started 11:01, all `ok`. The next step 51213347 waited on the folds (`afterany:51213345_*`) |
| `launch.json` written by the first step and read by the rest | Yes: written by 51213221 (`experiment 1 of 1, at most 40 steps`); `launches.jsonl` has that one line. Steps 51213347, 51244552 and 51254306 resumed from it without a new launch |
| DONE at the end | Yes: 51254306 exited 0 and wrote `DONE` at 16:40:59 |

**Can an agent reach Slurm?** Probed 11-05 on `sh04-17n29`, inside chain 0's allocation (`srun --overlap`, 1 CPU, 1 GB). The
probe ran a shell, with no agent, through the harness's own `sandbox_command` (as an opencode agent is wrapped, with no
credentials) and outside it for comparison (`stepmode_test/sandbox_sbatch_probe.py`):

- **Outside:** `squeue --me` works and `sbatch --test-only` validates a request.
- **Inside:** the binaries are visible (`/usr` is mounted) and no `SLURM_*` variable is set. Both commands fail Slurm's
  authentication: `failed to connect to any sack sockets` / `auth_g_create: REQUEST_CONFIG has authentication error`. Exit 1, no
  job created.
- **Why:** the sandbox mounts only `/usr`, `/etc` and `/share/software`, not `/run`, where Slurm's auth socket lives.
- **Caveats:** this is a property of the mounts, not a guarantee. It would change if `/run` were ever mounted, or if Slurm
  authentication stopped needing the socket. It was checked on one node only. A test that runs `squeue` in the sandbox and
  expects it to fail would catch that.

### Handoff §4 item 3: the real-agent segment (2026-11-06/07, code at 9d5b4252)

Written by Claude Opus 5.5 (the Sherlock session) from the jobs' logs, `sacct` and the run tree; not reviewed by a person.

**Bottom line: it worked.** Seven steps ran experiment 1 of 2, each agent session ran once, all 90 fit tasks completed, and
the last step wrote `STOP` and exited 15, which Slurm mailed to the PI. The only message from Slurm was that mail; SRCC sent
none (PI, 11-07). It took 21 h 47 m, of which the steps themselves ran 9 h 31 m. Nearly all of the rest was spent waiting
for fits, and only about 33 min in the queue.

**The launch.** The PI submitted it from a clone, `$SCRATCH/auto-psych/stepped_clone`, with
`WORK_ROOT=$SCRATCH/auto-psych/rsa_open_stepped_1`. The settings were `STEP_MODE=1 CHAIN=0 ARM=open N_EXPERIMENTS=2
MAX_ITERATIONS=2 CANDIDATES=3 STEP_PARTITION=normal STEP_FIT_PARTITION=owners`, with `sbatch --partition=normal
--cpus-per-task=8 --mem=32G --time=2-00:00:00 --no-requeue`. The steps ran on `normal` because chain 0 holds the lab node.
Two earlier submissions never reached step mode:

- **51286645.** The script's header `#SBATCH --array=0` made it a one-task array, which step mode refused. I cancelled it
  at 0 s, and 9d5b4252 fixed it.
- **51287612.** It stopped after 18 s with `FATAL: GOOGLE_API_KEY is not set (add it to .../stepped_clone/.secrets)`,
  because a fresh clone has no `.secrets` file. Copying the checkout's file (mode 600) fixed it. The handoff, or a check
  before submission, should say so.

**Steps** (8 CPUs and 32 GB each, all `COMPLETED` 0:0 except the stop):

| step | job | queue wait¹ | ran | MaxRSS | what it did | fits it left |
|---|---|---|---|---|---|---|
| 1 | 51328695 | 3.5 min | 0:53:36 | 5.8 GB | setup, starting set, ground truth | 15 (51333127) |
| 2 | 51333128 | 4.5 min | 2:10:23 | 7.4 GB | designer (1 session) and simulated pilot (726 calls) | 7 + 35 folds (51351904/05) |
| 3 | 51351906 | 3.5 min | 1:38:29 | **17.1 GB** | round 1: critique and 3 candidates | 3 (51393140, early) |
| 4 | 51393145 | 2.6 min | 1:02:02 | 9.0 GB | round 1: 2 repairs, admissions | 5 folds early (51402111), then 1 + 5 |
| 5 | 51406241 | 15.4 min | 2:35:29 | 13.6 GB | round 1 scored; round 2: critique and 3 candidates | 2 early (51423925), then 1 |
| 6 | 51428546 | 3.1 min | 1:03:59 | 7.9 GB | round 2: retry and repair, admissions | 10 folds early (51433936), then 1 + 5 |
| 7 | 51437423 | 3.8 min | 0:07:19 | 5.7 GB | round 2 scored, export, `STOP` | none |

¹ From the end of the last array the step waited on (or from submission, for step 1) to its start. The gaps between steps
(1 h 05 m to 5 h 22 m) were the fits themselves. The longest gap, before step 3, was 7 starting fits (4 h) followed by their
35 folds (1 h 05 m).

- **Fits:** 90 tasks across 12 arrays on `owners`, all `COMPLETED`, none preempted. Each took 2 min to 1 h 15 m. The largest
  MaxRSS was 4.8 GB, against the 18 GB tier. Arrays started 6-10 min after submission.
  - One candidate fit timed out at the admission limit: round 1 candidate 1's `quotient_partition_contrast`, after 71 min,
    with warm-up alone taking 55 min. It was rejected, and its repair was admitted.
  - Five fold tasks in 51351905 were skipped because their full fit wasn't in the cache. That is the rule for a failed model
    (the ledger's dropped seed `isolated_graded_costly_l3`), not an infrastructure error.
- **Memory with agents:** a step peaked at 17.1 GB, during round 1's critique and 3 concurrent candidates; 32 GB is ample at
  this size.
- **No session ran twice.** The token log counts 10 candidate sessions, which matches the 10 attempt directories and the 10
  ledger lines one for one:
  - round 1: candidates 1-3, plus repairs of 1 and 2
  - round 2: candidates 1-3, a retry of 1 (its first attempt wrote no `candidate.py`), and a repair of 2

  There were also 2 critiques, one per round, and 1 designer session. Steps 6 and 7 resumed "round 2 from its journal" without
  spawning anything again. `launches.jsonl` has one line.
- **The stop:** `[chain] STOPPED: experiment 1 of 2 is done; launch experiment 2`, then exit 15 (`FAILED` to Slurm, hence
  the mail). Experiment 1 exported `shared_prior_contrast_confusion_l1`. Tokens: 66.7 M, $27.59. The pilot's 2.98 M tokens
  have no price from the backend.
- **Messages:** none from SRCC. Slurm sent only the mail at the end. The agent CLI printed `Both GOOGLE_API_KEY and
  GEMINI_API_KEY are set. Using GOOGLE_API_KEY.` (harmless: the checkout's `.secrets` has both).

**For the driver:**

1. The `STOP` file's hint for the next launch is `CLEAR_STOP=1 STEP_MODE=1 (scripts/rsa/live/open_launch.sh)`. That is the
   **live** launcher, but this launch came from `open_rehearsal.sbatch`. The hint should name the script that made the launch.
2. A fresh clone needs `.secrets` (above).
3. Round times at this size were about 5.5-6 h. Round 1 ran from its critique at 16:10 to its last journal write at 21:20;
   round 2 from 21:48 to 03:16. Of that time, about half was spent waiting for fits. §12 uses this.

## 12. How long would a revised stepped chain take? (2026-11-07)

Written by Claude Opus 5.5 at the PI's request; not reviewed by a person. **This is an estimate.** It extrapolates from one
small segment (§11) to a size that has never run stepped.

**Bottom line: about 5-8 days for three full-size experiments, against about 10-11 days for chain 0.** Most of the saving
comes from failures that no longer cost a whole loop (about 2 days in chain 0). Rounds should also slow down less as the live
set and the data grow. Infrastructure won't make it much faster than that: the round count, the candidates per round and the
size of the live set decide how much fitting there is.

**Chain 0 so far** (launched 10-29 21:15 PT; at 11-07 07:00 it had run 8 days 10 h and is in experiment 3's round 3 of up to 5):

| | design + collection | model loop | losses |
|---|---|---|---|
| exp 1 | ~2.5 h | 38 h (5 rounds, ~7.6 h a round) | none |
| exp 2 | ~4.5 h | 72.3 h (35.4 h without failures; 5 rounds) | two out-of-memory crashes, restarts, waiting for fixes |
| exp 3 | ~7.5 h | 78 h so far | round 3 lost at the 48-h limit (~7 h) |

In experiment 3, the starting fits and standing took about 19.5 h. Round 1 was scored at 11-05 13:39 and round 2 at
11-06 13:04, so rounds took about 17-24 h, against about 7 h in experiment 1. On one node of 24 cores and 184 GB, 8-9 fits run
at once (§10). As the live set grew to 20-27 models and the data to three experiments, a round's fits queued behind each other,
and the slowest few ended it. Rounds 3-5, if all run, add about 1-3 days, for 10-11 days in all.

**What the segment says about stepped rounds:**

- The queue isn't the problem: about 33 min in all over 7 steps, even on `normal`.
- Fits on the critical path are: about 11.7 of 21.8 h. Arrays run every fit of a round at once, so what counts is the slowest
  fit, up to the admission limit, not how many there are.
- Each step reloads the loop. That was cheap at 7-9 live models. Chain 0 measured 35-74 min at 20-27 (§2, §10), on about 3
  steps a round.

**Estimate per full-size experiment** (6 candidates, 20+ live models, cumulative data):

- **Design and collection: 4-8 h.** Collection is about 1.5-2.5 h; design was 3-5 h in chain 0, plus PI review when the
  paradigm is new.
- **Starting fits and folds on the new data: 3-6 h.** That is 5.3 h in the segment and 19.5 h in chain 0's experiment 3, on
  one node.
- **Each round: 6-10 h.** The agents run concurrently, so 6 candidates take about as long as 3. Add the slowest fit, the
  repairs and their fits, the folds, and 2-4 h of restarts at 20+ live models.
- **Five rounds: 30-50 h**, fewer if two stale rounds stop the loop early. Chain 0 never stopped early.
- **Relaunch latency: hours.** Step mode stops after each experiment by design, and a person relaunches. A closed laptop or a
  day of teaching adds directly.

That is about 40-65 h per experiment, or 5-8 days for three.

**What could move it:**

- **Up:** the slowest candidate fits. The admission limit is 4 h, and one slow candidate holds its round's next step for that
  long. Full-size rounds have never run stepped.
- **Down:** steps on `mcfrank` once chain 0 is done (no queue, and no other jobs on the node). Saving CV scores to disk would
  remove most of a restart's cost (§9, A1). Folds that reuse the full fit's adaptation (B1, which needs an amendment) and a
  compile step (A5) would also help.
- **The design itself:** rounds per experiment, candidates per round and the live-set cap decide how much fitting there is.
  Gate item 5's limit of 104 h of running time is already past in chain 0 by any count. If the revision registers a new
  limit, set it against these parameters as well as the infrastructure.

## 13. Fit speed with the pattern-weighted likelihood (`HANDOFF_fit_speed.md`, 2026-11-07, code at 786e609b)

Written by Claude Opus 5.5 (the Sherlock session); not reviewed by a person. No agents, no Prolific, nothing written to
chain 0; the corpus copy and every trial stay on Sherlock (`$SCRATCH/auto-psych/fitspeed/`). One task was still running when
this was written (marked below).

**Bottom line: on the committed literature the new code is 30× faster, but on chain 0's experiment 3 data it is only
2.5-2.9× faster, and it needs no less memory.** A typical admission fit goes from about 2.5 h to about 1 h, not to minutes,
and still peaks at 36-39 GB. The bug the change fixes was real: on a cheap theory nearly all of the time was the per-trial
overhead. The agents' theories are expensive to compute on each display, though, and for them that computation is now most
of the cost. I think most of what's left is in that computation and in compiling, and neither is measured yet (below).

### How it was run

- Both codes on the same CPU (AMD EPYC 8224P, `CPU_GEN:SIE`, the lab node's chip), on `owners`/`normal`, each fit pinned
  to one core with `taskset` and XLA/BLAS set to one thread. The old code is `rsa-open-prereg-a2` (12051f8, what chain 0 runs)
  with the handoff's `fit_speed.py` copied in and its data line cut to trial counts (the old `OpenData` has no patterns).
- 4 chains (vectorized) × 1,000 warm-up + 1,000 draws, dense mass, the loop's defaults. The compile cache was off, so
  **warm-up includes compiling**, as in the handoff's own numbers.
- Theories: the depth-3 seed `isolated_graded_costly_l3` (102 lines), and experiment 3's three slowest admitted theories
  with one full-data sidecar each (482-492 lines each, 17 parameters). `cue_claim_figure_anchor_heuristic` was slower but
  has two sidecars and comes from the round still running, so I left it out.
- Data: experiment 3's cumulative corpus, the one its loop fits (copied, read-only):

| block | trials | displays | patterns | display shapes |
|---|---|---|---|---|
| E (literature) | 50,587 | 260 | 587 | 10 |
| exp 1 | 2,952 | 24 | 679 | 4 |
| exp 2 | 4,530 | 29 | 578 | 4 |
| exp 3 | 4,650 | 29 | 1,779 | 5 |
| **all** | 62,719 | 342 (327 distinct across blocks) | 3,623 | 23 (13 across blocks) |

  The committed literature alone (handoff item 1) is 27,949 trials, 173 displays and 402 patterns.

### Results (same CPU, one core; every fit converged)

| theory | data | chain 0's recorded time | old code here | new code | speedup | new warm-up / sampling | steps/iter, old → new | ms per leapfrog step, old → new | peak memory, old → new |
|---|---|---|---|---|---|---|---|---|---|
| depth-3 seed | literature | — | 58.9 min | **2.0 min** | **30×** | 71 s / 32 s | 14.6 → 14.3 | 118 → 3.6 | 2.5 → 2.4 GB |
| `figure_ground_crowd_topic_tom` | exp 3 | 156.0 min | 150.2 min | **52.1 min** | **2.9×** | 29.3 / 21.5 min | 17.6 → 15.4 | 249 → 99 | 34.6 → 36.0 GB |
| `cloned_anchor_partition_shield_tom` | exp 3 | 148.9 min | 148.3 min | **59.7 min** | **2.5×** | 33.2 / 25.1 min | 16.4 → 15.6 | 264 → 112 | 37.3 → 39.1 GB |
| `fuzzy_intensional_truth_tom` | exp 3 | 147.2 min | 143.0 min | **54.8 min** | **2.6×** | 30.9 / 22.6 min | 16.1 → 15.2 | 259 → 106 | 34.5 → 36.4 GB |
| depth-3 seed | exp 3 | — (dropped before exp 3) | *running, > 2 h 50 min* | **42.2 min** | ? | 36.4 / 4.8 min | ? → **67.3** | ? → 18 | ? → 5.9 GB |

"Steps/iter" is leapfrog steps per iteration, warm-up and sampling together; with vectorized chains each iteration costs as
much as its longest chain, so that is what's counted. "ms per leapfrog step" is (warm-up + sampling time) / steps, so it
includes compiling in both codes. Building the data took 26 s → 2.8 s (literature) and 54-57 s → 3.7-4.1 s (experiment 3).
Peak memory is `/usr/bin/time`'s maximum resident set size.

### What the numbers say

1. **The old code's per-trial overhead is gone.** On the seed, per-step cost fell 33×, which matches the driver's
   measurement on a cloud container (25.9 → 1.8 ms per gradient; 119 s for the fit on 4 cores, against 118 s here on one
   EPYC core).
2. **On the agents' theories, per-step cost fell only about 2.4×** (249-264 → 99-112 ms). The rest of the speedup comes from
   taking slightly fewer steps (for `figure_ground`, 15.4 against 17.6 per iteration). The two codes compute the same density,
   so I read the step difference as a different trajectory from rounding, not a real change. About 100 ms per step for 342
   displays is roughly 0.3 ms per display per 4-chain step, against about 0.02 ms for the seed. **So the theory itself is
   now most of the cost.** That is an inference from the totals: the displays differ in size and I haven't profiled anything.
3. **Chain 0's recorded times match the old code run alone** (148.9 vs 148.3 min, 147.2 vs 143.0, 156.0 vs 150.2). Load on
   the lab node did not slow chain 0's full fits much. That is unlike the folds in §10, which ran 2-3× slower in the chain.
4. **Memory didn't fall; it rose by about 1.5-2 GB.** On the first try, at the handoff's 24 GB, all three big theories were
   killed for running out of memory 7-12 minutes into the fit, at about 25 GB. The peak comes early, which fits §10's finding
   that compiling takes most of a fit's memory. **24 GB is too little for these theories; 48 GB is safe.**
5. **The seed is hard to sample on experiment 3's data:** 67 steps per iteration against 14 on the literature, and 36 of its
   42 minutes in warm-up. Its sampling phase was short (4.8 min), so most of those steps were probably in warm-up, while the
   step size and mass matrix were still adapting. That is geometry, not per-step cost. The old run will say whether the old
   code saw the same.
6. **I couldn't separate compile time from these timings.** In both codes warm-up costs *less* per step than sampling: in
   chain 0's sidecars for `cloned_anchor`, 263 against 291 ms (21,060 and 11,648 steps). If compiling took, say, 10 minutes,
   warm-up should cost more per step, not less. So either compiling is short for these theories, or something else makes
   sampling steps dearer. I don't know which.

### Where value is probably still left (hypotheses, ranked; none tested)

1. **The theories' own computation.** It is now probably 85% or more of a step: the seed's whole step on the same data,
   links on all 3,623 patterns included, costs about 18 ms of the agents' 99-112. Nothing else in the fit can save more
   than the remaining share. A profile of one theory's gradient would tell us where it goes: compile time, and per-evaluation time for the
   whole density, the theory alone and the link alone, by display shape, forward and backward. That is the question for the
   memo developer: whether memo's generated code for these roughly 490-line theories (nested ToM with partitions over
   candidate sets) does avoidable work, for example recomputing inner inferences that don't depend on the outer variable,
   or running in float64. I can write and run such a profile (about 30 min a theory, on two theories) if the driver wants it.
2. **Compiling, for memory more than for time.** If the 36-39 GB peak is compiling, a compile step of its own (A5) and a
   persistent cache that the full fit and its 5 folds share would cut each fit's peak to a fraction. Memory, not cores,
   limits how many fits the lab node can run at once: about 4-5 at 36-39 GB in 192 GB, which leaves most of its 24 cores
   idle. A direct measurement would settle it: compile time and peak memory alone, then a cached fit's peak.
3. **One theory call per shape across blocks.** Each block evaluates its own displays, in 23 shape groups. Merged, that is 13
   groups and 327 displays instead of 342. Under 5% of theory time, and perhaps a smaller graph to compile. Small.
4. **Sampler geometry for the hard cases** (item 5 above): only worth it if such theories are common. The agents' theories
   here take about 15 steps per iteration, which is already good.
5. **Chains on separate cores** buy wall time, not core-hours (§10 B2 measured little gain on the old code; untested on the
   new one).

### What it means for a centralized chain (rough; not measured beyond the full fits)

Each admitted theory costs a full fit plus 5 folds. On the old code, chain 0's folds took about 90-107 min each (on the lab
node, under load) and full fits about 2.5 h, so about 10 core-hours a theory. If the folds speed up as much as the full fits
did (not tested), the new code needs about 4-5. Memory per fit is unchanged, though, so on one node the number of fits at once
is still about 4-5, set by memory. So the change helps a centralized chain by about 2.5× in fitting time, not by an order of
magnitude, unless (1) or (2) above also pays off.

### Not done or not as specified

- **The slow tests** that the handoff asks for first (added in 0c15ba4b, after these fits started) are running now (job
  51484554, `normal`/`owners`, on the 786e609b clone).
- I did not compare old and new posteriors on experiment 3's data: `fit_speed.py` prints only timing and convergence.
  The driver's tests compare the likelihoods.
- The first try (job 51473074) ran the new code on Intel Broadwell (Xeon E5-2640 v4) nodes: the seed on the literature took
  197 s there against 118 s on EPYC. I cancelled the rest of that array and reran all five new-code tasks on EPYC at 64 GB
  (51476315). The old code (51473075, 48 GB) was on EPYC throughout.
- Compute: about 3.6 core-hours for the new code, about 8.4 plus the running task for the old, and about 1 for the first try.
  Memory requests bought each job 2-9 allocated CPUs, but each fit ran on one of them.

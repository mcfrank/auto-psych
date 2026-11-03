# Open chain 1 (`chain_0`): status for the driver, 2026-11-02 22:00 PST

For the PI's strategy discussion with the driver. Experiment 1's report is
`RUN_CHAIN1_EXP1_REPORT.md`; experiment 2's full report follows when its loop ends.

## The short version

- **The chain is running and healthy** on tag `rsa-open-prereg-a1` (Addendum B, posted on OSF by the PI): experiment
  2's loop resumed after round 2 and is scoring round 3. No fit has been killed since the fix.
- **The PI decided tonight** (2026-11-02): experiment 3 runs; item 5's running time is reported, not judged (Addendum C,
  committed in 12051f8, not yet posted on OSF); the scheduler's bug gets fixed.
- **In progress:** tests of that fix (job 50875639), then tag `rsa-open-prereg-a2`, then a restart just after a round is
  scored.
- **The PI's question for the driver: how to parallelize.** A month of babysitting sequential jobs isn't feasible.
  What's measured for that is under "Parallelizing stage 2" below.

## Where the time went

The registered item 5 counted from experiment 1's design start to experiment 3's loop end, less each collection stage,
with "queue waits and failures" included. Under Addendum C the same figure is reported, not judged.

| period | hours | counted | note |
|---|---:|---:|---|
| experiment 1 design | 2.8 | 2.8 | |
| PI review, deploy, collection | 5.8 | – | the collection stage |
| experiment 1 loop | 38.1 | 38.1 | 5 rounds; registered estimate 20 h |
| experiment 2 design | 3.5 | 3.5 | with the gap before it |
| collection | 1.3 | – | |
| loop attempt 1 | 15.2 | 15.2 | **lost**: out of memory at 96 GB; restarted from scratch |
| stopped | 5.4 | 5.4 | **lost**: unnoticed for 5 h (my poll ran late), then the decision |
| loop attempt 2 | 19.0 | 19.0 | rounds 1-2 kept (12.1 h); **round 3 lost** (6.9 h): out of memory at 180 GB |
| stopped | 9.4 | 9.4 | **lost**: writing and testing the fix |
| loop attempt 3, so far | 3.9 | 3.9 | |
| **total** | | **97.2** | **about 37 h lost to the two failures** |

**Projection** (rough; rounds so far took 5-7 h): about 150 h as the code stands, about 140 h with decision 2's fix; less
if a loop stops early on two stale rounds. I told the PI about 133 h earlier; these replace that.

**The estimate itself was low**, failures aside. §3a assumed a loop of about 20 h (about 3 rounds); every loop so far has
a new leader each round, so all 5 rounds run (38 h in experiment 1). Without the failures, and with the fixes throughout,
a chain would take very roughly 90-105 h. That is an estimate, not a measurement: it's right at the limit, not clearly
under it.

## Spend

| | agent spend |
|---|---:|
| experiment 1 | $109.52 |
| experiment 2 design | $2.87 |
| experiment 2 loop attempt 1 (lost) | $43.58 |
| experiment 2 loop attempt 2 (about a third lost) | $60.11 |
| experiment 2 loop attempt 3, so far | $17.41 |
| priced total | $233.49 |
| unpriced pilot calls, estimated | $27.15 |
| **total, as the gate counts it** | **$260.64 of $495** |

Projected at the end of 3 experiments: about $435, under the limit (a rough figure). Prolific: $265.07 for experiment 1,
about $265 for experiment 2; balance $2,590.59. The $109.52 in experiment 1's report was priced calls only; with its
share of the estimated pilot calls it's about $128.

## What failed and what changed

The scoring step ran 4 theories x 5 folds at once with no limit on memory, beside a parent process that held about 0.9 GB
per live theory. Addendum B: every fit waits for room in the job's memory; a killed fit is retried once; fits load
without the per-trial log-likelihood; candidate fits overlap retry agents and CV starts at admission; a loop resumes from
its last scored round. No setting, seed, cache key, gate or score changes. Tests: 109 pass for the loop and open campaign;
the fast suite's 13 failures aren't in the changed code (10 are packages missing from the shared environment; 3 are
subjective-randomness scripts, 2 of them an old-bash error and 1 not diagnosed).

**A bug in my fix, found tonight (being fixed).** It's safe but slow. The scheduler reads each fit's peak from `ru_maxrss`, which in a
freshly spawned process carries over the parent's size (16 GB). So every fit records 15.8 GB and is reserved about 20 GB.
Measured on the node at 22:00: **7 fits running, real peaks 6.7-10.5 GB, 96 GB free, 17 CPUs idle.** Using `VmHWM` is a
one-line fix, and should allow about 12-15 fits at once.

**Measured memory**, for planning: a fit takes 7-10.5 GB, and the loop's own process 16.6 GB with 15 live theories. So
memory, not CPU, limits a 24-CPU node at 184 GB.

## Experiment 2 so far

Paradigm `pragmatic_choice_and_probed_rating` (the designer changed paradigm), 4.02 bits, 0.0152 bits per dollar.
Study filled in 79 min: 163 recruited, 151 kept (7% excluded, against 13% in experiment 1), median 6.1 min, $11.91/h,
enjoyment 3.7 of 5. Its blocklist was verified to name experiment 1's study.

Loop: the leader changed in rounds 1 and 2 (`prototype_figure_individuated_mixture_l2`, then
`polar_figure_ground_topic_tom`); 15 live theories after round 2. Every slot's candidate was admitted in those two rounds except one
slot that wrote no file. The critique's significant statistics are falling: 6, 5, then 3 of 8 for round 3.

## Decided by the PI, 2026-11-02

1. **Experiment 3 runs.** Its design starts by itself when experiment 2's loop ends (about midday to evening on 11-03),
   and its study (about $265) publishes about 3 h later.
2. **The scheduler's peak gets fixed** (`VmHWM`; tag `rsa-open-prereg-a2`). Saves about 2 h per remaining round, up to
   about 14 h.
3. **Item 5's running time is reported, not judged; agent spend stays judged at $495** (Addendum C, `gate.py`). Addendum C
   says plainly that this came after the outcome was foreseeable: at 97.2 of 104 h, on time and spend only, before
   experiment 3. Without the amendment, §4 would have made chain 1 a pilot and required a revision. The PI still has to
   post it on OSF.

## Parallelizing stage 2 (the question for the driver)

What's registered: open chains 2-4 and fixed chains 1-4, 7 chains of 3 or 4 experiments, scale set at the gate.

**Measured here:**
- One chain now takes a whole lab node: 24 CPUs and about 184 GB. The lab has one such node (`mcfrank`: 192 GB, 24 CPUs),
  so one after another 7 chains would take about 4 weeks at about 100 h each.
- **Memory limits it, not CPU:** a fit takes 7-10.5 GB, and the loop's own process 16.6 GB with 15 live theories.
- **A round's fits are the bulk of its compute:** 6 candidate fits (60-80 min each) and about 30 fold fits (40-60 min
  each), single-threaded. Agents use all of their 60 minutes.
- **Studies are not the constraint:** they filled in 79 and 120 min, and `max_concurrent_studies` is already 8.

**Where chains could run:**

| partition | fits a chain? | catch |
|---|---|---|
| `mcfrank` | yes, one chain | one node |
| `normal` | yes (8 GB per core, nodes of 192 GB or more) | long queue, up to 7 days a job |
| `owners` | yes (up to 48 GB per core) | preemptible; with resume, a preemption costs at most one round |
| `bigmem` | yes | 1 day a job; workable now that a loop resumes across jobs, but the queue's small |

**What changes the picture:**
- *Resume makes short and preemptible jobs usable.* Before, a loop had to finish within one job. So `owners` is now
  reasonable, with the launcher's ladder of jobs picking up each interruption.
- *A failure should not need a person:* a failed job writes STOP and stops the ladder (§4a's "a person looks"). That's
  where the babysitting comes from, and it was also why a crash sat unnoticed for 5 h. The driver could consider
  automatic resume for infrastructure failures with a cap on attempts, and a notification.
- *Sherlock's rules:* no unattended agent sessions and no jobs that resubmit themselves (`/etc/agents/AGENTS.md`), so
  supervision needs job dependencies, as the ladder uses.
- *Smaller fits, if a revision allows it:* E is 50,587 of the 58,069 trials in a full fit. That would be a change of
  method, not an infrastructure fix. So would fewer rounds, shorter theorist sessions or a lower admission limit (2 x the
  median starting fit, which rose from 73 to 125 min between experiments).

**Not measured yet:** how the scheduler does at 12-15 fits at once (that comes with a2), whether `owners` preempts often
in practice over days (job 1 ran 38 h there without it), and how the shrinking participant pool affects later studies.

## Corrections to record

- **Addendum B says the two failures "cost about 33 h of loop time" and that each restarted the loop from scratch.** The
  second kept rounds 1-2, because the resume landed before the restart. Measured: about 22 h of loop time lost and about
  15 h stopped, 37 h in all. Addendum C corrects this.
- Experiment 1's report gave $109.52 as the chain's agent spend; that excludes the estimated pilot calls (above).

## Running

Jobs 50852526 (running, lab node, 24 CPUs, 184 GB) → 50852527 → 50852528 → 50852529. Experiment 2's two abandoned
attempts are kept as `loop.abandoned_1` and `round_3_abandoned_1`. The next check is on round 3's scoring, the step that
crashed before.

# Open chain 1 (`chain_0`): status for the driver, 2026-11-02 22:00 PST

For the PI's strategy discussion with the driver. Experiment 1's report is
`RUN_CHAIN1_EXP1_REPORT.md`; experiment 2's full report follows when its loop ends.

## The short version

- **The chain is running and healthy** on tag `rsa-open-prereg-a1` (Addendum B, posted on OSF by the PI): experiment
  2's loop resumed after round 2 and is scoring round 3. No fit has been killed since the fix.
- **The gate's time criterion will fail.** Item 5 allows 104 h. The chain is at 97.2 h and passes 104 h at about 05:00
  PST on 11-03, before experiment 2 ends. It should finish at about 140-150 h.
- **Under §4 that decides a lot:** "no further chains run under this registration. A revision is registered anew, and
  open chain 1 is then reported as a pilot, not counted."
- **The next decision is time-sensitive.** Experiment 3's design starts by itself when experiment 2's loop ends (about
  midday to evening on 11-03), and its study (about $265) publishes about 3 h later. Decision 1 below should be made
  before then.

## Where the time went

Item 5 counts from experiment 1's design start to experiment 3's loop end, less each collection stage; "queue waits and
failures count".

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

**A bug in my fix, found tonight.** It's safe but slow. The scheduler reads each fit's peak from `ru_maxrss`, which in a
freshly spawned process carries over the parent's size (16 GB). So every fit records 15.8 GB and is reserved about 20 GB.
Measured on the node at 22:00: **7 fits running, real peaks 6.7-10.5 GB, 96 GB free, 17 CPUs idle.** Using `VmHWM` is a
one-line fix, and would allow about 12-15 fits at once. See decision 2.

**Measured memory**, for planning: a fit takes 7-10.5 GB, and the loop's own process 16.6 GB with 15 live theories. So
memory, not CPU, limits a 24-CPU node at 184 GB.

## Experiment 2 so far

Paradigm `pragmatic_choice_and_probed_rating` (the designer changed paradigm), 4.02 bits, 0.0152 bits per dollar.
Study filled in 79 min: 163 recruited, 151 kept (7% excluded, against 13% in experiment 1), median 6.1 min, $11.91/h,
enjoyment 3.7 of 5. Its blocklist was verified to name experiment 1's study.

Loop: the leader changed in rounds 1 and 2 (`prototype_figure_individuated_mixture_l2`, then
`polar_figure_ground_topic_tom`); 15 live theories after round 2. Every slot's candidate was admitted in those two rounds except one
slot that wrote no file. The critique's significant statistics are falling: 6, 5, then 3 of 8 for round 3.

## Decisions for the PI and driver

**1. Does experiment 3 run?** (Before experiment 3's design starts, about midday to evening on 11-03.)
- *Run it:* about $265 and about $125 of agents, and about 40 more hours. The gate's other four items get measured, and
  the pilot is complete. Those items can't be computed on 2 experiments.
- *Stop after experiment 2:* saves that money and time, and the revision can be registered sooner, but items 1-4 are
  known only for 2 experiments, and the third round of the design-from-data cycle goes unseen.
- *My recommendation:* run it unless the revision needs that money. The chain's purpose is now to be a good pilot, and a
  complete 3-experiment record is what the revision's estimates would rest on. To stop, I'd write the STOP file before
  the design starts.

**2. Fix the scheduler's peak?** It needs a tag `rsa-open-prereg-a2`, a line added to Addendum B, and a restart. I'd
restart just after a round is scored, so almost nothing is lost. It saves about 2 h per remaining round, up to about 14
h. *Recommendation: yes*; it's only worth skipping if decision 1 is to stop.

**3. What happens to item 5?** As registered, it fails and chain 1 becomes a pilot. The alternative is an amendment
changing the limit or excluding failures, but the text says failures count, and the outcome is already known, so that
would be a post-hoc change to a criterion. *My recommendation: let it fail as written*, and put realistic estimates in
the revision. If the PI prefers to keep chain 1 counted, it should be labelled as a deviation and decided now, before
experiment 3's data, using time and spend only (as §4 requires of scale decisions). That's the PI's call.

**4. Can stage 2 run on this hardware?** One chain takes the lab node whole (24 CPUs, about 184 GB), so 7 more chains one
after another would take about 4 weeks of that node. In parallel they need other nodes: `normal` fits this size (8 GB per core, on its nodes of 192 GB or more) and isn't
preemptible, but its queue is long; `owners` is preemptible, and with resume a preemption now costs at most one round;
`bigmem` allows only 1 day a job. Smaller fits would change this, and the corpus is the place to look: E is 50,587 of the
58,069 trials in a full fit. But changing that
would be a change of method for the revision, not an infrastructure fix.

**5. Revision levers**, if a revision is coming: the time estimates (5 rounds, not 3); round count or the stale-round
rule; the 60-minute theorist session (agents always use all of it); and the admission limit, which is 2 x the median
starting fit and rose from 73 to 125 min between experiments.

## Corrections to record

- **Addendum B says the two failures "cost about 33 h of loop time" and that each restarted the loop from scratch.** The
  second kept rounds 1-2, because the resume landed before the restart. Measured: about 22 h of loop time lost and about
  15 h stopped, 37 h in all. Worth correcting in the line that decision 2 adds.
- Experiment 1's report gave $109.52 as the chain's agent spend; that excludes the estimated pilot calls (above).

## Running

Jobs 50852526 (running, lab node, 24 CPUs, 184 GB) → 50852527 → 50852528 → 50852529. Experiment 2's two abandoned
attempts are kept as `loop.abandoned_1` and `round_3_abandoned_1`. The next check is on round 3's scoring, the step that
crashed before.

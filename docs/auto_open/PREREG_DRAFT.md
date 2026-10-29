# Open and fixed-paradigm discovery loops: draft pre-registration

**Status: final text for registration (2026-10-29), not yet registered.** Registered on OSF
**before the feasibility chain (§4) starts**, so the gate is fixed before any data that it judges,
together with the git tag `rsa-open-prereg` of the code every live chain runs (the live launcher
refuses any other: `scripts/rsa/live/open_live.yaml`, `code_tag`) and the sha256 of the files that
decide the gate (§8). Two registered amendments follow: the scale of stage 2, set at the gate
(§4), and the bakeoff's analysis (§6, marked **TBD**), before any bakeoff participant is recruited,
as `rsa-prereg-final` was for the closed campaign (`docs/auto_rsa/PREREG_FINAL.md`). The plan it
implements is `PLAN.md` (Decisions 2026-10-26 to 2026-10-29).

## 1. Question and claims

Do discovery loops that design their own paradigms produce theories that generalize across
operationalizations better than loops that refine theories on one fixed paradigm? This is
Yarkoni's (2022) generalizability problem: a theory fitted to one task licenses no claim about
the construct.

1. **Transfer.** On a blind paradigm, the open arm's final theories predict new people better than
   the fixed arm's.
2. **Parsimony.** At equal or better transfer, they use fewer theory parameters, and their
   parameters are more stable across paradigms (each paradigm fitted alone, estimates compared).
3. **The designer.** Agent-designed paradigms are valid and informative: realized against
   predicted information gain, comprehension and catch pass rates, pilot against live, and how
   participants rated them (an end-of-study "How enjoyable was this study?", 1-5, on every page of
   both arms). Claim 3 is descriptive; its measures are fixed in §7.

## 2. Two arms, one harness

Both arms run the same code (`src/rsa_open/`), and differ only in **who chooses each experiment**:
- **Open arm:** an agent designer proposes a new paradigm each experiment (`PLAN.md` §4). The
  harness checks it, scores it, pilots it and freezes it. A designed paradigm uses at least 3
  domains of pictures (or text materials) and at most 30 test trials a person; the designer's
  brief asks it to weigh variety and the effort of each answer, and it reads earlier experiments'
  enjoyment ratings (PI 2026-10-29). Its pay is at least $10 an hour at the larger of its
  declared minutes and the harness's time estimate, which is calibrated on this campaign's own
  completion times (run D2's two studies from the start; PI 2026-10-29).
- **Fixed arm:** every experiment uses the closed campaign's forced-choice paradigm (one frozen
  spec). Its displays are chosen by the same information-gain tool, with no designer.

Everything else is identical by construction:
- the theory format and the links;
- fitting, cross-validation, selection, the 4-SE guard and pruning, with each earlier paradigm's
  best theory kept as its reference;
- the inference limits: every fit is limited to 4 hours, and a theory whose fit or cross-validation
  does not converge (or runs out of time) is not carried into the next experiment (the
  incumbent stays if no theory converges);
- the agents: designer, theorists and critic are `google/gemini-4-argon` (Gemini 4), the pilot's
  simulated participants `gemini-3.8-flash`; the sandbox (no network), rounds and slots, and
  theorist tooling (the sampler report, the identification prompt, the corpus-relative admission
  limit, the quick approximate fit);
- the seeds and the chain split, and the literature data E;
- the per-experiment budget (§3);
- the page builder and recruitment.

**Up to four chains per arm, each of up to 4 experiments** (PI 2026-10-26); the scale actually run
is set at the gate (§4). Chain k of each arm starts from the same seeds; chain 4 repeats chain 1's
(the closed campaign's chain 0 seeds), so one pair of chains measures chain-to-chain variance from
one start.
After the feasibility chain, the two arms' chains run concurrently, their studies recruiting in
parallel (PI 2026-10-29), so time, Prolific's pool and the agents' model versions are shared. A
person takes part at most once in each chain (§4b). Each study's listing uses one neutral name and
description for both arms, with the paradigm's own pay and minutes.

The closed campaign (October 2026) is reported as history and pilot evidence. It is not an arm:
it ran older code and agents, and its pre-registered data are spent.

## 3. Resources per experiment

**$266.67 per experiment, for every experiment of both arms** (PI 2026-10-26): twice the closed
campaign's $133.33 (200 people × $0.50 × Prolific's fee). The open designer spends it as it
chooses, within the bounds; the fixed arm spends it on the closed paradigm's design, with more
people at the same pay and length: 400 people × $0.50 for 3 minutes (its time estimate is 2.8-2.9
minutes for any design, and run D2's people took a median of 2.9).

## 3a. Estimates (the gate's reference, item 5)

From the rehearsals (runs B, B′, C and C′), per experiment at full size:
- **time:** the design 2–3 h and the loop about 20 h (starting fits 4–5 h, then about 3 rounds of
  5–6 h): **23 h of running** an experiment, which the gate's time criterion uses (§4, item 5); and
  collection, 1–24 h (24 h at most, §4a; run D2's 15 people a study took about two hours, at
  night), which it does not;
- **money:** the budget's $266.67 (§3), and **at most $110 of agent spend** (from $55).

So the feasibility chain's first 3 experiments: about 3–4 days of running (waits for the PI's
review excluded), $800 of participant payments and at most $330 of agent spend. On a pass, open
chains 2–4 and fixed chains 1–4 at full scale: about 1–1.5 weeks, about $7,470 of participant
payments and $1,700–3,000 of agent spend.

## 4. The feasibility gate

**Open chain 1 runs first and alone**, live. The gate judges its first 3 experiments; its 4th runs
only if the gate passes and stage 2 keeps 4 experiments per chain (below). Its procedure is the registered
one, and its data count as the open arm's chain 1 if the gate passes. The gate is about whether
the method can run with real people. It looks at no comparison with the fixed arm, since there is
none yet.

**Pass** requires all of these (thresholds accepted by the PI, 2026-10-26; definitions,
2026-10-29). They are computed by `src/rsa_open/gate.py` (`python -m src.rsa_open.gate --run-dir
<the chain>`) from the chain's run tree alone, and that script is registered (§8):
1. **Design:** each experiment freezes a paradigm that passes the pilot, within the designer's two
   cycles (a harness submission counts).
2. **Data:** in each experiment, among recruited participants: final comprehension ≥ 0.8, catch
   trials passed ≥ 0.8, exclusions (§4b) ≤ 25%, the data parse (the experiment's block is written
   and its loop runs on it), and at most 10% of readable participants give the same answer on
   nearly every test trial (the monitor's flag: at least 90% of their test answers the same).
3. **Information:** each experiment's data carry signal its theories capture: the loop's
   incumbent at its first scoring (the best of the theories carried in, before any new one) beats
   a **chance theory** on the new paradigm by more than 4 unit-clustered SEs, both scored by the
   loop's grouped cross-validation. The chance theory ignores the display (uniform over its objects;
   its derived speaker uniform over the messages) and its link is fitted, so it predicts each
   paradigm's distribution of answers with no sensitivity to the scene. How far apart the carried
   theories are is reported, not judged: the loop prunes them to near-ties, and in run B's
   simulated chain they were 0.4 and 1.0 SEs apart in experiments 2 and 3 although the chain worked.
4. **Loop:** each experiment's loop ends with at least 2 trusted theories and an export (no
   harness failure), and **over the 3 experiments** at least one new theory is admitted (run B's
   experiment 3 admitted none: its loop stopped after two rounds without a change of incumbent).
5. **Cost and time:** within 1.5× the estimate (§3a): at most 104 h of running for the 3
   experiments (1.5 × 3 × 23 h), counted from experiment 1's design start to experiment 3's loop
   finish less each collection stage (recruitment, the deploy, and the wait for the PI's review;
   queue waits and failures count), from the chain's stage log; and at most $495 of agent spend
   (1.5 × 3 × $110; every priced call plus an estimate for unpriced ones at their model's rate).
   Participant payments are bounded by the budget by construction, and are reported.

**If it passes:** open chains 2–4 and fixed chains 1–4 run as registered. **If it fails:** no
further chains run under this registration. A revision is registered anew, and open chain 1 is
then reported as a pilot, not counted.

**The scale of stage 2 is set at the gate** (PI 2026-10-29), from the feasibility chain's cost and
time only:
- **chains per arm:** 3 or 4 (with 3, chain 4, the repeat of chain 1's seeds, is not run);
- **experiments per chain:** 3 or 4, the same for every chain of both arms (with 3, open chain 1's
  final theory is its third experiment's incumbent, and its 4th is not run).

The decision is recorded in a registered amendment before any stage-2 study is published. It
uses the feasibility chain's time and spend against §3a, never its theories' scores, and it binds
both arms alike. The bakeoff's unit of inference stays the chain.

## 4a. Failures (accepted by the PI, 2026-10-26)

The rule is fixed in advance and is the same for every chain, so no chain is rerun because of
how it is going.
- **Infrastructure** (a crash, a timeout, a node lost, an API outage): the chain resumes from its
  last finished stage. Nothing is discarded.
- **A designer failure before any live data** (both cycles fail in experiment 1): the chain is
  restarted once, from the same seeds, with a fresh designer. Nothing has been spent or seen, so
  the restart selects on nothing but the designer's luck. The failure is reported (claim 3).
- **A designer failure after live data** (experiment n > 1): the chain stops there. Its last
  finished experiment's incumbent is its final; it is not rerun or replaced, and is reported as
  a shorter chain. Rerunning it would select chains on how their designs went.
- **An underfilled study:** a study recruits for at most 24 hours (from its publication, through
  any restart of its job). It is then paused, and the partial data are modelled.
- **The fixed arm** cannot fail at design; its infrastructure failures resume like any other.

## 4b. Participants (accepted by the PI, 2026-10-26; reuse, 2026-10-29)

- **A person takes part at most once in each chain** (PI 2026-10-29). Each study's Prolific
  blocklist names its own chain's earlier studies, so a chain's experiments are independent
  samples and each committed model is scored on people new to that chain (claim 2). People may
  take part in other chains, of either arm, and people from the closed RSA campaign and the
  subjective-randomness studies may take part. The campaign needs roughly 14,000 responses;
  excluding everyone who had done any RSA study asked more of Prolific's pool than it holds (run
  D2: 15 people took about two hours to recruit). Taking one referent game can change how a
  person approaches the next, and a returning person in the fixed arm repeats nearly the same
  task, so this is measured (next item) rather than assumed away.
- **Returning people are recorded and analysed.** Every study a person takes part in is recorded
  privately (`src.rsa_open.live.campaign_ids`), and each study reports how many of its people had
  taken part in earlier studies of this campaign, and in how many (aggregates). **Sensitivity
  analysis:** every pre-registered result is recomputed on each person's first study of the
  campaign only, and both are reported. The closed campaign's people cannot be identified as
  returning (its ids are not in this campaign's record), which the report states.
- **Exclusions (accepted by the PI, 2026-10-26):** a participant's data are left out if
  they missed any catch trial (the closed campaign's rule), failed the comprehension checks on the
  second reading, had taken part in an earlier experiment of the same chain (a backstop to the
  blocklist), or left an incomplete record. Nothing else: no exclusions by time or by answer
  pattern. People who give the same answer on nearly every trial are monitored and reported, not
  excluded. The gate's "exclusions ≤ 25%" (§4) counts these exclusions.
- About 32 experiments × 400 people plus the bakeoff is roughly 14,000 responses. Each study
  recruits for up to 24 hours (§4a; the closed campaign's studies filled in about an hour, against
  a 3-hour window; run D2's took about two). Recruitment speed is monitored and reported. The
  exclusions are never relaxed mid-campaign.

## 5. Competitors in the bakeoff

- Each chain's final incumbent: 4 open, 4 fixed.
- The seeds, as a baseline.
- The closed campaign's three finals, as history: reported, not tested.

## 6. The bakeoff (analysis TBD in the second registration)

- **Arms:**
  - **F, forced choice:** the fixed paradigm on fresh displays.
  - **T, transfer:** a blind paradigm with a new response format, designed by an isolated designer
    run that no chain or theorist has seen. Half its displays are chosen for information and half
    sampled at random from the pool, so generalization is over stimuli as well as tasks.
- **Fits:**
  - **A (primary):** every competitor fitted on the same pooled data (E plus both arms' live data).
    This compares theories.
  - **B (secondary):** each competitor fitted on its own arm's data. This compares methods,
    including the data each chose to collect.
- **Predictions:** the open finals beat the fixed finals in T. In F, the fixed finals do at least
  as well.
- **Unit of inference for the method claim:** the chain (4 per arm). Theory comparisons use
  participants. **TBD:** the score, the tests and thresholds, power and sample sizes, exclusions,
  and the frozen pool predictions with their hashes.
- **Links in T** are fitted on as little of T's data as the analysis allows, and claim 2's
  stability is reported for every competitor, so that a flexible link cannot pass for a general
  theory.

## 7. Also reported (descriptive, not tested)

- Each arm's admission outcomes: too slow, not converged, admitted.
- **Claim 3's measures**, for every experiment, by arm, summarized over chains (each chain one
  unit: the mean of its experiments, and the range across chains):
  - **information:** the design's predicted joint information gain (bits, `design/frozen/record.json`)
    against the realized gain: log2 K minus the entropy of the posterior over the K theories
    carried in, uniform before and proportional to exp(their out-of-fold lpd on the new paradigm)
    after (`loop/newest_block_cv.json`); the carried theories' spread on the new paradigm, and the
    incumbent's lead over chance (§4, item 3);
  - **validity:** final comprehension, catch pass rate, exclusions by reason, readable share, and
    the share giving the same answer on nearly every trial;
  - **pilot against live:** comprehension, catch pass rate and minutes, the pilot's against real
    people's;
  - **participants' experience:** the enjoyment rating's mean and distribution (comments stay
    private), and the median minutes and pay per hour at the median;
  - **the designer's record:** cycles used, harness submissions, failures (§4a), and the domains,
    response types and test trials it chose.
- Returning people per study and arm (§4b), and the first-study sensitivity analysis beside every
  result.
- The rehearsals: run B, its replay and the ground-truth check, and the integrated harness's
  rehearsal (run C, simulated) and live smoke test (run D: about 15 people per arm on a
  rehearsal's design, never modelled).
- **Scope:** every paradigm shares one abstract scene (referents, features, a message, its
  alternatives). Generalization is claimed within that space of operationalizations.

## 8. Registered files

The tag `rsa-open-prereg` fixes all the code, and a live chain runs only on it (its checkout at the
tag, clean). These files decide the gate or the fixed arm; their sha256 are given here and on OSF
so the registration can be checked without the repository (computed at tagging):

| file | sha256 |
|---|---|
| `src/rsa_open/gate.py` | (at tagging) |
| `src/rsa_open/fixed_paradigm.json` | (at tagging) |
| `src/rsa_open/loop/orchestrator.py` (the chance theory, the scores the gate reads) | (at tagging) |

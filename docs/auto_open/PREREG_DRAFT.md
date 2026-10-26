# Open and fixed-paradigm discovery loops: draft pre-registration

**Status: draft, not registered.** Registered on OSF **before the feasibility chain (§4) starts**,
so the gate is fixed before any data that it judges. A second, registered amendment fixes the
bakeoff's analysis (§6, marked **TBD**) before any bakeoff participant is recruited, as
`rsa-prereg-final` was for the closed campaign (`docs/auto_rsa/PREREG_FINAL.md`). The plan it
implements is `PLAN.md` (Decisions 2026-10-26).

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
   predicted information gain, comprehension and catch pass rates, pilot against live.

## 2. Two arms, one harness

Both arms run the same code (`src/rsa_open/`), and differ only in **who chooses each experiment**:
- **Open arm:** an agent designer proposes a new paradigm each experiment (`PLAN.md` §4). The
  harness checks it, scores it, pilots it and freezes it.
- **Fixed arm:** every experiment uses the closed campaign's forced-choice paradigm (one frozen
  spec). Its displays are chosen by the same information-gain tool, with no designer.

Everything else is identical by construction:
- the theory format and the links;
- fitting, cross-validation, selection, the 4-SE guard and pruning, with each earlier paradigm's
  best theory kept as its reference;
- the theorist and critic (Gemini 4 for every agent), the sandbox, rounds and slots, and theorist
  tooling (the sampler report, the identification prompt, the corpus-relative admission limit);
- the seeds and the chain split, and the literature data E;
- the per-experiment budget (§3);
- the page builder and recruitment.

**Four chains per arm**, each of 3 experiments (depth under discussion, 2026-10-26). Chain k of
each arm starts from the same seeds; chain 4 repeats chain 1's (the closed campaign's chain 0
seeds), so one pair of chains measures chain-to-chain variance from one start.
After the feasibility chain, the two arms' chains run concurrently, with recruitment interleaved,
so time, Prolific's pool and the agents' model versions are shared. One study recruits at a time
across the campaign, in the order the chains are ready, so Prolific's blocklist (applied when a
study is published) covers every earlier study. Every study excludes everyone who took part in
any earlier RSA study (§4b). Each study's listing uses one neutral name and description for both
arms, with the paradigm's own pay and minutes.

The closed campaign (October 2026) is reported as history and pilot evidence. It is not an arm:
it ran older code and agents, and its pre-registered data are spent.

## 3. Resources per experiment

**$266.67 per experiment, for every experiment of both arms** (PI 2026-10-26): twice the closed
campaign's $133.33 (200 people × $0.50 × Prolific's fee). The open designer spends it as it
chooses, within the bounds; the fixed arm spends it on the closed paradigm's design, with more
people at the same pay and length.

## 4. The feasibility gate

**Open chain 1 runs first and alone**, live, all 3 experiments. Its procedure is the registered
one, and its data count as the open arm's chain 1 if the gate passes. The gate is about whether
the method can run with real people. It looks at no comparison with the fixed arm, since there is
none yet.

**Pass** requires all of these (thresholds accepted by the PI, 2026-10-26):
1. **Design:** each experiment freezes a paradigm that passes the pilot, within the designer's two
   cycles (a harness submission counts).
2. **Data:** in each experiment, among recruited participants: final comprehension ≥ 0.8, catch
   trials passed ≥ 0.8, exclusions ≤ 25%, and the data parse and pass the monitor's degeneracy
   checks.
3. **Information:** each experiment's data separate its design theories: on the new paradigm, the
   best and worst design theories differ by more than 4 clustered SEs (held-out).
4. **Loop:** each experiment's loop admits at least 1 new theory, ends with at least 2
   trusted theories, and finishes without a harness failure.
5. **Cost and time:** within 1.5× the plan's estimate.

**If it passes:** open chains 2–4 and fixed chains 1–4 run as registered. **If it fails:** no
further chains run under this registration. A revision is registered anew, and open chain 1 is
then reported as a pilot, not counted.

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
- **An underfilled study** (Prolific's poll ends short): the partial data are modelled, as now.
- **The fixed arm** cannot fail at design; its infrastructure failures resume like any other.

## 4b. Participants (accepted by the PI, 2026-10-26)

- **No one takes part twice within this campaign**, in either arm. Taking one referent game
  changes how a person approaches the next, and the arms must not share people.
- **People from earlier auto-psych studies:**
  - those from the closed RSA campaign are excluded (the same task);
  - those from the subjective-randomness studies are **allowed** (a different task), which
    widens the pool.
- **Exclusions (proposed, 2026-10-26; awaiting the PI):** a participant's data are left out if
  they missed any catch trial (the closed campaign's rule), failed the comprehension checks on the
  second reading, had taken part in an earlier study of the campaign (a backstop to the
  blocklist), or left an incomplete record. Nothing else: no exclusions by time or by answer
  pattern. People who give the same answer on nearly every trial are monitored and reported, not
  excluded. The gate's "exclusions ≤ 25%" (§4) counts these exclusions.
- About 32 experiments × 400 people plus the bakeoff is roughly 14,000 people. Recruitment speed
  is monitored. If a study's fill rate falls below half the first experiment's, its collection
  window is extended; the exclusions are never relaxed mid-campaign.

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
- Claim 3's designer measures, pilot against live.
- The rehearsals: run B, its replay and the ground-truth check, and the integrated harness's
  rehearsal (run C, simulated) and live smoke test (run D: about 15 people per arm on a
  rehearsal's design, never modelled).
- **Scope:** every paradigm shares one abstract scene (referents, features, a message, its
  alternatives). Generalization is claimed within that space of operationalizations.

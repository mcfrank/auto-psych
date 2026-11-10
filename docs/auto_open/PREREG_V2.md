# Open and fixed-paradigm discovery loops: pre-registration, revision 2 (draft)

**Status: a draft for the PI's review, 2026-11-10 (the PI's decisions of 11-10 included). Not registered.** Written by Claude Opus 5.5; not
reviewed by a person. It is the revision that §4 of the first registration (https://osf.io/78t69,
2026-10-29, `PREREG_DRAFT.md`) calls for: "a revision is registered anew, and open chain 1 is then
reported as a pilot, not counted." Its sections follow the first registration's numbering, so the two
can be read side by side; the appendix lists every change and its reason.

> **Decided by the PI on 2026-11-10:** 4 chains per arm × 3 experiments × 3 rounds of 6; the stage-1 pair
> counts as chain 1 of each arm when the gate passes; no agent-spend criterion (spend is reported); the
> prospective test is reported, not claimed; chain 1 is a pilot. **Still to fill before registering:** chain 1's
> final state from Sherlock (§0), and the sha256 of the registered files at tagging (§8).

## 0. Why a new registration, and what exists already

- **Open chain 1 (the code's `chain_0`) is a pilot** (PI 2026-11-08). It ran under the first registration,
  live, for 11 days of clock time: two experiments finished; a third recruited its participants and its
  loop started, ⟨to confirm from Sherlock: how far it ran⟩. Its data, theories, timings and spend are reported
  as the pilot (§7) and never counted in the bakeoff.
- **What the pilot showed, which this revision acts on:**
  - **Slow fits.** The open fit evaluated every trial; a large theory's fit took 2.5 h and 35-37 GB, and the
    chain ran more than 8 days for 3 experiments. Fits now compute the same density about 10× faster (appendix).
  - **Hand-set numbers.** Chain 1's experiment-1 theories carried 9 hand-tuned constants each (median; up to
    27) beside 2-5 fitted parameters, and 32 of those 35 would be refused under §2's new rule; its exported
    theories set 17, 24 and 27 by hand with 2-3 parameters (review V3 §16). A number
    set by looking at all the data was tuned on the scenes cross-validation holds out too.
  - These findings came from looking at chain 1's theories, timings and scores. The changes they motivated
    (§2, appendix) are about how theories are written and chosen, not about which arm should win.
- **What does not exist yet:** no chain of the fixed arm has run live, no stage-2 chain has run, and no
  bakeoff paradigm has been designed. A stepped rehearsal of both arms (simulated participants, no Prolific;
  review V3 §16) and no-agent replays (§17) tested the code this revision registers.

## 1. Question and claims (unchanged except claim 2's note)

Do discovery loops that design their own paradigms produce theories that generalize across
operationalizations better than loops that refine theories on one fixed paradigm? (Yarkoni, 2022.)

1. **Transfer.** On a blind paradigm, the open arm's final theories predict new people better than the
   fixed arm's.
2. **Parsimony.** At equal or better transfer, they use fewer theory parameters, and their parameters are
   more stable across paradigms. *New note:* both arms' theorists are now told that short theories are the
   aim, and both arms choose the incumbent by the one-SE rule (§2), so parsimony is measured twice: the final
   incumbent's (which the rule shapes) and that of the best theory by fit alone at the same step (which it
   does not). A difference between arms in either is a result; the shared steering may make both smaller.
3. **The designer.** Agent-designed paradigms are valid and informative (descriptive; measures in §7).

## 2. Two arms, one harness

The arms run the same code and differ only in **who chooses each experiment** (open: an agent designer;
fixed: the closed campaign's forced-choice paradigm, its displays chosen by the same information-gain tool),
as in the first registration. Identical by construction, as before, and now also:

- **No hand-set numbers** (PI 2026-11-08). A theory's numbers outside its priors are 0, 1, `EPS` (a
  tolerance or floor) and `BIG` (a mask value), and integers that index arrays or give their sizes. Any other
  literal is refused at admission and by the self-check (`src/rsa_open/loop/gates.py`,
  `magic_number_problems`). The seeds were rewritten to pass, with identical predictions.
- **The one-SE rule** (PI 2026-11-08). Of the trusted theories eligible under the 4-SE guard, those within
  **1 clustered SE** of the best on the newest paradigm, the one with the **fewest fitted parameters** leads
  (ties: the better fit) (`src/rsa_open/corpus.py`, `simplest_adequate`). Pruning still measures distance from
  the best fit and never prunes the incumbent. Each scored step records the best by fit alone (`fit_best`).
- **The theorists' brief** (`src/rsa_open/loop/prompts/open_theory.md`) states the aim (a short theory, with
  few parameters that each mean something about the person, that carries over to new people and paradigms),
  the one-SE rule, and that simplifying is progress; a refinement's one change may be a simplification; each
  theory's standing shows its parameter count and why the incumbent leads.
- **Fitting:** each distinct display is evaluated once and each response pattern once; the sampler reaches the
  density through a compiled callback. The density, priors, NUTS settings (4 chains × 1,000 + 1,000, dense
  mass), convergence gate and near-miss refit are unchanged. Every fit is limited to 4 h, as before.
- **Agents:** designer, theorists and critic Gemini 4 (`google/gemini-4-argon`), the pilot's simulated people
  Gemini 3.8 Flash; sessions of 60 minutes; the sandbox with no network.

**Scale** (PI 2026-11-10): 4 chains per arm, each of **3 experiments**; each experiment's loop runs **3 rounds of 6**
theorists (3 explore, 2 refine the incumbent, 1 refines a theory of its choice). Chain k of
each arm starts from the same seeds; chain 4 repeats chain 1's seeds, as before.

**How a chain runs** (infrastructure; it changes no setting, score or decision): one launch per chain runs
every experiment (`LAUNCH_EXPERIMENTS=all`) as one job on the lab node, which runs its own fits on its cores
and continues from disk after each phase (`STEP_FITS=local`); a job that runs short of time is continued by a
new one. A slow test holds a loop run this way to one run straight through: the same ledger, history and
export, each agent once (`tests/test_rsa_open_step_equivalence.py`).

## 3. Resources per experiment (unchanged)

$266.67 per experiment of both arms: the open designer spends it within its bounds; the fixed arm on 400
people × $0.50 for 3 minutes.

## 3a. Estimates (replacing the first registration's)

From the stepped rehearsal (review V3 §16: one chain per arm, experiment 1, 3 rounds of 6, simulated people):
- **Running time:** 7.7 h (fixed) and 14.3 h (open) for experiment 1; projected 27-31 h and 48-53 h for 3
  experiments, before recruitment (1-24 h a study). Fits now run inside the step's job; the no-agent replays
  ran faster than with arrays (§17), so these projections are, if anything, high.
- **Agent spend:** about $37 (fixed) and $74 (open) for experiment 1, both understated (some calls unpriced).
  Reported for every experiment; not a criterion (PI 2026-11-10).
- **Participants:** the budget, $266.67 an experiment: $800 a chain, $6,400 for 8 chains.

## 4. The feasibility gate

**Stage 1: open chain 1 and fixed chain 1 run together**, live, each for 3 experiments. The gate judges both,
on feasibility only: it looks at no comparison between them.

**Pass** requires every item, for **each** of the two chains (computed by `src/rsa_open/gate.py` from the
chains' run trees, `python -m src.rsa_open.gate --run-dir <open chain 1> <fixed chain 1>`; registered, §8):
1. **Design:** each experiment freezes a paradigm: for the open chain one that passes the pilot within the
   designer's two cycles; for the fixed chain its frozen spec, through the harness's checks.
2. **Data:** in each experiment, among recruited participants: final comprehension ≥ 0.8, catch trials passed
   ≥ 0.8, exclusions ≤ 25%, the data parse, and at most 10% of readable participants give the same answer on
   at least 90% of test trials. (As before.)
3. **Information:** in each experiment, the incumbent at its first scoring beats the chance theory on the new
   paradigm by more than 4 unit-clustered SEs. (As before.)
4. **Loop:** each experiment's loop ends with at least 2 trusted theories and an export, and at least one new
   theory is admitted over the 3 experiments. (As before.)
5. **Time** (new): each experiment runs within **48 h** (PI 2026-11-09), counted from its design's start to
   its loop's export, less recruitment (queue waits and failures count). Agent spend is reported, not judged
   (PI 2026-11-10).

**If both pass:** the pair counts as chain 1 of each arm (PI 2026-11-10), and chains 2-4 of both arms run,
concurrently.
**If either fails:** no further chains run under this registration; a revision is registered anew, and the pair
is reported as a pilot. Either way the gate's outcome and every item's value are reported.

## 4a. Failures, 4b. Participants (unchanged)

As in the first registration, with one addition: since a chain runs as one launch, an infrastructure failure is
resumed by relaunching the chain, which continues from its last finished phase. The blocklist (a person at most
once per chain), returning-people analysis and exclusions stand as registered.

## 5. Competitors in the bakeoff

Each chain's final incumbent (4 open, 4 fixed), the seeds as a baseline, and as history (reported, not tested)
the closed campaign's three finals and the pilot chain 1's final.

## 6. The bakeoff (unchanged: analysis TBD in its own registration, before any bakeoff data)

## 7. Also reported (descriptive, not tested)

As in the first registration (admission outcomes; claim 3's measures; returning people and the first-study
sensitivity analysis; the rehearsals), and:
- **Parsimony, per theory and arm:** fitted parameters, lines of code, the incumbent and `fit_best` at every
  scored step with their parameters, how often the one-SE rule changed the incumbent, refinements that removed
  parameters, and candidates refused for hand-set numbers.
- **The prospective test** (PI 2026-11-10): each experiment's committed theory scored on the next experiment's people, whom
  no agent saw, against its cross-validated score, as a check on cross-validation.
- **The pilot:** open chain 1's experiments (two finished, a third begun), theories with their hand-set
  constants, timings, participant and agent spend.

## 8. Registered files

A new annotated tag, `rsa-open-prereg-v2`, fixes all the code; live chains run only on it. The sha256 of the
files that decide the gate, the fixed arm and selection, computed at tagging:

| file | sha256 |
|---|---|
| `src/rsa_open/gate.py` | at tagging |
| `src/rsa_open/fixed_paradigm.json` | at tagging (unchanged from the first registration's, expected) |
| `src/rsa_open/loop/orchestrator.py` | at tagging |
| `src/rsa_open/corpus.py` (the one-SE rule, the guard) | at tagging |
| `src/rsa_open/loop/gates.py` (no hand-set numbers) | at tagging |
| `src/rsa_open/loop/prompts/open_theory.md` (the theorists' brief) | at tagging |

## 9. AI disclosure (unchanged)

Drafted by Claude Opus 5.5 (Anthropic) working with Michael C. Frank, who made every decision recorded here and
is responsible for it. The designer, theorists and critic under study are Gemini 4 models, the pilot's simulated
people Gemini 3.8 Flash.

## Appendix: changes since the first registration

| change | when, who | reason | what it changes |
|---|---|---|---|
| Loop scheduling and resume (Addendum B) | 11-02, PI | chain 1's experiment 2 lost its loop twice to memory | infrastructure |
| Running time reported, not judged (Addendum C) | 11-02, PI | the 104-h limit measured harness failures | a gate criterion, stated as such then |
| Fits: each display and pattern once; compiled callback; one start search per chain | 11-08 | fits 2.5 h → about 15 min for a large theory | infrastructure: same density and posterior (tests) |
| Fits on one core (two XLA pool threads); fits free their compiled programs | 11-08 | a deadlock and a memory leak found on Sherlock | infrastructure |
| Chain 1 is a pilot | 11-08, PI | slow-fit decisions and hand-set constants | §0, §4 |
| No hand-set numbers | 11-08, PI | §0 | which theories are admitted |
| The one-SE rule; the theorists' brief on parsimony | 11-08, PI | theories grew a component a round; short theories transfer | which theory leads; what theorists aim for |
| Stage 1: an open and a fixed chain together | 11-08, PI | test both arms' feasibility before stage 2 | §4 |
| Fits in the step's own job; one launch per chain | 11-09, PI | 316 of 325 fit tasks ran under 10 min (rule 10) | infrastructure (equivalence test) |
| Gate item 5: 48 h per experiment; agent spend reported, not judged | 11-09 and 11-10, PI | measured times (§3a); spend is not a hypothesis | a gate criterion |
| The stage-1 pair counts if the gate passes | 11-10, PI | as open chain 1 would have | §4 |
| Scale: 4 chains × 3 experiments × 3 rounds of 6, fixed now | 11-10, PI | measured times | scale (was set at the gate) |

# OSF registration of the open campaign

The plan is `PREREG_DRAFT.md` (final text, 2026-10-29). This file is how it is frozen on OSF: the
steps, and the answers for OSF's **"OSF Preregistration"** template, section by section, as for the
closed campaign (`docs/auto_rsa/OSF_PREREG.md`). Each answer is short and points to the plan.

**OSF takes no uploaded files**, so the registration links a **tagged commit** of the public
repository: `rsa-open-prereg`, https://github.com/mcfrank/auto-psych/tree/rsa-open-prereg. It quotes
the tag's full commit SHA (fixed by the content of every file in the commit, so the link stays
verifiable even if a tag moves) and the sha256 of the files that decide the gate ("Other").

**This is the first of three registrations.** It fixes the method, both arms, the feasibility gate
and the participant rules, before any live data of the campaign. Two registered amendments follow:
the scale of stage 2, set at the gate from cost and time only, before any stage-2 study; and the
bakeoff's analysis, before any bakeoff participant is recruited.

## Steps

1. **The freeze** (driver): `code_tag: rsa-open-prereg` in `scripts/rsa/live/open_live.yaml`, the
   hashes in `PREREG_DRAFT.md` §8 and below; commit; tag that commit `rsa-open-prereg` (annotated);
   push the branch and the tag; give the PI the tag's full commit SHA (the tagged file cannot hold
   its own commit's SHA).
2. **The registration** (PI, on osf.io: a registration cannot be withdrawn):
   1. Create a project, "auto-psych: open and fixed-paradigm discovery loops" (or a component of
      the auto-psych project).
   2. **Registrations → New registration → "OSF Preregistration"**; paste the answers below, with
      the commit SHA in "Other".
   3. Make it public (not embargoed), submit, and approve it from the confirmation email: OSF makes
      it final only after that approval, or after 48 hours.
3. **Record it** (driver): the registration's URL, its approval date and the commit SHA go in
   `PREREG_DRAFT.md` as Addendum A and in `PLAN.md`.
   - **Open chain 1's live launch waits for this.** Its test deploy (page and Prolific draft, for
     the PI's review of its first paradigm) does not: it publishes nothing.
   - Sherlock re-stages at the tag; the live launcher refuses any other code.

## Answers for the template

### Study information

**Title:** Do discovery loops that design their own experiments find more general theories of
pragmatic reference than loops that refine theories on one fixed experiment?

**Authors:** Michael C. Frank (and collaborators, as on the paper).

**Description:**
- **The method:** auto-psych runs automated discovery chains. In each experiment, coding agents
  (Gemini 4) propose computational theories of how people interpret or choose a speaker's message
  in simple reference games (rational-speech-act models and alternatives); every theory is fitted
  to all the chain's data with NUTS and selected by grouped cross-validation; the next experiment
  is designed to tell the surviving theories apart; real people on Prolific take it.
- **Two arms**, identical except for who chooses each experiment:
  - **open:** an agent designer proposes a new paradigm each experiment (a new response format,
    new materials, new scenes), which the harness checks, scores for expected information gain,
    pilots with simulated participants and freezes;
  - **fixed:** every experiment uses one frozen forced-choice paradigm (the earlier closed
    campaign's); only its displays are chosen, by the same information-gain tool.
- **The question** (Yarkoni's generalizability problem): do the open arm's final theories
  generalize across operationalizations better than the fixed arm's?
- **Stages:** (1) a feasibility chain: open chain 1 alone, live, judged by a registered gate;
  (2) on a pass, 3–4 chains per arm of 3–4 experiments each; (3) a bakeoff on a blind paradigm.
  This registration fixes stages 1 and 2's procedures; the bakeoff's analysis is a later amendment.
- **The record:** `docs/auto_open/PREREG_DRAFT.md` and the code at the tag `rsa-open-prereg` of
  https://github.com/mcfrank/auto-psych; the plan's decisions and reasons are in
  `docs/auto_open/PLAN.md`.

**Hypotheses:**
- **H1 (transfer):** on a blind paradigm, the open arm's final theories predict new people better
  than the fixed arm's. On the fixed paradigm, the fixed arm's do at least as well.
- **H2 (parsimony):** at equal or better transfer, the open arm's finals use fewer theory
  parameters, and their parameters are more stable across paradigms.
- **H3 (the designer), descriptive:** agent-designed paradigms are valid and informative (measures
  fixed in the plan, §7).
- H1 and H2 are tested in the bakeoff; their scores, tests, thresholds and sample sizes are fixed by
  the bakeoff amendment, before any bakeoff data. **Confirmatory now: the feasibility gate**, a
  pass/fail decision on open chain 1's first 3 experiments (Analysis plan).

### Design plan

**Study type:** Experiment: a sequence of online experiments, each designed from the data before
it. The between-chain comparison of the arms is the study's design; nothing is manipulated within
participants beyond what each paradigm varies.

**Blinding:**
- Participants do not know the hypotheses or the arm (one neutral study listing for both arms).
- The agents (designer, theorists, critic) have no internet and never see held-out data, other
  chains' data or theories (designers read only other chains' completion times and enjoyment
  ratings, as aggregates), or the earlier campaign's live or pre-registered data or final theories.
- The gate is a fixed script applied to the chain's run tree.

**Study design:**
- **Chains:** up to 4 per arm, each of up to 4 experiments; chain k of each arm starts from the
  same seed theories (5–6 per chain, from the earlier campaign's starting sets), and chain 4 repeats
  chain 1's seeds, to measure chain-to-chain variance from one start. The literature data the
  earlier campaign started from (50,587 trials) are every chain's first block.
- **Each experiment:** design (the open designer's paradigm, or the fixed paradigm's displays) →
  live collection → the inner loop (up to 5 rounds of theorist agents, admission gates, grouped
  cross-validation within each paradigm, a 4-SE guard on every earlier paradigm, pruning) → the
  live set carried to the next experiment.
- **Bounds on a designed paradigm:** at least 100 participants, at most 15 minutes and 30 test
  trials each, at least 3 picture domains, at least $10 an hour at the larger of the declared and
  the estimated time, scenes of 2–6 objects; no deception or affectively charged material; the PI
  reviews each open chain's first paradigm before it is published.
- **The fixed paradigm:** 400 people, 13 designed displays each (of 52), 2 catch trials and a
  practice trial, $0.50 for 3 minutes.
- **The budget:** $266.67 per experiment in both arms.
- Details: `PREREG_DRAFT.md` §2–3; `PLAN.md`.

**Randomization:**
- Each participant gets one of the paradigm's trial lists, in arrival order; the lists are balanced
  subsets of the design (every display answered equally often, to within one).
- Within a list, trial order and the objects' screen positions are randomly permuted, by a seed
  fixed at the design's freeze.
- After the feasibility chain, the two arms' chains run concurrently, recruiting in parallel.

### Sampling plan

**Existing data:** **Registration prior to creation of data** for this campaign. Its chains start
from published literature data (fitted by the seed theories before this registration), and the
rehearsals' smoke tests (30 people, never modelled) gave only completion times.

**Explanation of existing data:** the literature data are every chain's first block and were used
to build and fit the seed theories. No data of this campaign's chains exist at registration. The
earlier closed campaign's live and pre-registered data are not used by either arm (only its
completion times, as aggregates, calibrate the time estimate, and its paradigm is the fixed arm's).

**Data collection procedures:**
- **Recruitment:** Prolific, desktop only, approval rate ≥ 98%.
- **Participants:** a person takes part at most once in each chain (each study's Prolific
  blocklist names its chain's earlier studies); people may take part in other chains, of either
  arm, and in the earlier campaign. Every study a person takes part in is recorded privately.
- **Pay:** each paradigm's own, at least $10 an hour (the fixed arm: $0.50 for 3 minutes).
- **Consent:** the consent text approved under the existing IRB protocol.

**Sample size:**
- **Per experiment:** the open designer's N (≥ 100, within the budget); the fixed arm's 400.
- **Experiments:** the feasibility chain's 3 (4 if stage 2 keeps 4 per chain); on a pass, 3–4
  chains per arm × 3–4 experiments per chain, set at the gate and registered in an amendment.

**Sample size rationale:** the per-experiment budget, twice the earlier campaign's, fixes the
spend; the scale is limited by cost and time (`PREREG_DRAFT.md` §3a). The unit of inference for the
arms is the chain. Power for the bakeoff is set in its amendment.

**Stopping rule:**
- A study recruits until its N submissions or for 24 hours from publication (across restarts),
  then is paused and its partial data modelled.
- A chain stops after its last experiment, or at a designer failure after live data (the chain is
  then reported as shorter; never rerun). A designer failure before any live data restarts the
  chain once from the same seeds (`PREREG_DRAFT.md` §4a).
- If the gate fails, no further chain runs under this registration.

### Variables

**Manipulated variables:** the arm (open or fixed), between chains. Within the open arm, each
designer chooses its paradigm (response format, materials, scenes); within each paradigm, the
displays are chosen for information.

**Measured variables:**
- each trial's response in the paradigm's format (a choice, a rating, a slider, a bet over the
  objects, a message produced, a yes/no judgment);
- comprehension checks, catch trials, completion time, and an end-of-study enjoyment rating (1–5;
  an optional comment is kept private).

**Indices:**
- each theory's out-of-fold log predictive density on each paradigm (grouped cross-validation,
  folds of whole units), and differences with unit-clustered standard errors;
- the gate's indices (Analysis plan); claim 3's measures (`PREREG_DRAFT.md` §7).

### Analysis plan

**Statistical models:**
- **Fitting and selection** (every experiment, both arms): each theory and each paradigm's link
  fitted jointly by NUTS to all the chain's data; ranking by out-of-fold lpd on the newest
  paradigm; eligibility within 4 clustered SEs of the best on every earlier paradigm; a fit
  limited to 4 hours; a theory whose fit or folds do not converge is not carried.
- **The feasibility gate** (confirmatory now), computed by `src/rsa_open/gate.py` on open chain 1's
  experiments 1–3. It passes when all hold:
  1. each experiment froze a paradigm within the designer's two cycles;
  2. in each, final comprehension ≥ 0.8, catch trials passed ≥ 0.8, exclusions ≤ 25%, the data
     parse, and ≤ 10% of readable participants give the same answer on ≥ 90% of test trials;
  3. in each, the loop's incumbent at its first scoring beats a chance theory (no sensitivity to the
     display, its link fitted) on the new paradigm by more than 4 unit-clustered SEs, both by the
     loop's grouped cross-validation;
  4. each loop ends with ≥ 2 trusted theories and an export, and ≥ 1 new theory is admitted over
     the 3 experiments;
  5. running time ≤ 104 h (design and loops; collection and the PI's review excluded) and agent
     spend ≤ $495.
- **The bakeoff:** fixed by its amendment (`PREREG_DRAFT.md` §6 outlines it: every competitor
  fitted on pooled data, scored on a fixed-paradigm arm and on a blind transfer paradigm).

**Transformations:** none. Catch and practice trials are not modelled.

**Inference criteria:** the gate passes only if every criterion holds (no correction: it is one
decision). On a pass, stage 2 runs; its scale (3 or 4 chains per arm, 3 or 4 experiments per
chain, the same for both arms) is chosen from the feasibility chain's time and spend only, never
its theories' scores, and registered before any stage-2 study. On a failure, no further chain runs;
a revision is registered anew and open chain 1 is reported as a pilot.

**Data exclusion:** a participant's data are left out if they missed any catch trial, failed the
comprehension checks on the second reading, had taken part in an earlier experiment of the same
chain, or left an incomplete record. Nothing else: no exclusion by time or answer pattern.

**Missing data:** only submitted, complete sessions are analysed; Prolific returns and timeouts
never reach the data. An underfilled study's partial data are modelled.

**Exploratory analysis** (reported, not tested):
- claim 3's measures: predicted against realized information, validity, pilot against live,
  enjoyment and pay, the designer's record;
- each arm's admission outcomes; how far apart the carried theories are on each new paradigm;
- returning participants per study and arm, and every result recomputed on each person's first
  study of the campaign;
- the rehearsals (simulated runs and the 30-person smoke tests).

### Other

**Deviations:** any deviation is reported with its reason, in a dated addendum to
`PREREG_DRAFT.md` and in the paper.

**AI disclosure:** The text of this pre-registration (`PREREG_DRAFT.md` and `OSF_PREREG.md`) was drafted by Claude Opus 5.5 (Anthropic), an AI model, working with Michael C. Frank in Claude Code, which also wrote much of the campaign's code (its commits name it as co-author). Michael C. Frank made every design decision recorded here (`PLAN.md` dates each one), reviewed and approved the text, and is responsible for it. Separately, AI agents are part of the method under study: the designer, theorists and critic are Gemini 4 models, and the pilot's simulated participants Gemini 3.8 Flash (Study information).

**The frozen materials:** the git tag `rsa-open-prereg` of https://github.com/mcfrank/auto-psych,
commit `<full SHA, given at tagging>`.

| file (in that commit) | sha256 |
|---|---|
| `src/rsa_open/gate.py` (the gate) | `e63c476346883f753902a4871ec0162fe44b47edfdaf439586cfcf83c4acc5ce` |
| `src/rsa_open/fixed_paradigm.json` (the fixed arm's paradigm) | `98247195c5755fcee16ebfe7f65f04c599f224c8774b3c8a6e52c004ba0cdda1` |
| `src/rsa_open/loop/orchestrator.py` (the loop; the chance theory and the scores the gate reads) | `684c1d5a179ba92e8436ca04e6bdfdedef5ea09afc2ee933cce5e91a8045dd36` |

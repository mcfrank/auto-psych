# OSF registration of the open campaign, revision 2

The plan is `PREREG_V2.md`. This file is how it is frozen on OSF: the steps, and the answers for OSF's
**"OSF Preregistration"** template, section by section, as for the first registration (`OSF_PREREG.md`).
Each answer is short and points to the plan.

**It is a new registration, linked to the first** (https://osf.io/78t69), which said that if its feasibility
chain did not count, "a revision is registered anew, and open chain 1 is then reported as a pilot." OSF takes
no uploaded files, so the registration links a **tagged commit** of the public repository,
`rsa-open-prereg-v2`, quoting its full commit SHA and the sha256 of the registered files ("Other").

**Registered: https://osf.io/wurh5 (2026-11-13), tag `rsa-open-prereg-v2`, commit `b32e732b74d99d5f3dd85a9374d7d2e6b44beab8`** (`PREREG_V2.md`, Addendum A).

## Steps

1. **The freeze** (driver, done 2026-11-11): the registered scale and run mode as the live defaults, `code_tag:
   rsa-open-prereg-v2` in `scripts/rsa/live/open_live.yaml`; the fast suite passes; commit `b32e732b74d99d5f3dd85a9374d7d2e6b44beab8` on
   `auto-psych-open`; its files' sha256 in `PREREG_V2.md` §8 and "Other" below (a later commit: a file cannot hold
   its own commit's SHA).
2. **The tag** (PI: the cloud session can push its branch but not tags). In a checkout of the repository:
   ```bash
   git fetch origin auto-psych-open
   git tag -a rsa-open-prereg-v2 b32e732b74d99d5f3dd85a9374d7d2e6b44beab8 -m "Open campaign, revision 2 (docs/auto_open/PREREG_V2.md): the code every registered live chain runs."
   git push origin rsa-open-prereg-v2
   git rev-parse 'rsa-open-prereg-v2^{commit}'   # must print b32e732b74d99d5f3dd85a9374d7d2e6b44beab8
   ```
3. **The registration** (PI, on osf.io; a registration cannot be withdrawn):
   1. In the project of the first registration, **Registrations → New registration → "OSF
      Preregistration"**; paste the answers below, with the commit SHA and hashes in "Other".
   2. Link the first registration (https://osf.io/78t69) in its description, as below.
   3. Make it public, submit, and approve it from the confirmation email (final after approval, or 48 hours).
4. **Record it** (driver): the URL, approval date and commit SHA go in `PREREG_V2.md` (an addendum) and
   `PLAN.md`. **The stage-1 launch waits for this.** Sherlock stages the tag; the live launcher refuses any
   other code.

## Answers for the template

### Study information

**Title:** Do discovery loops that design their own experiments find more general theories of pragmatic
reference than loops that refine theories on one fixed experiment? (Revision 2)

**Authors:** Michael C. Frank (and collaborators, as on the paper).

**Description:**
- **This is a revision of https://osf.io/78t69 (2026-10-29), registered anew as its §4 requires.** Its
  feasibility chain (open chain 1) ran live for 11 days, finished two experiments and began a third, and is
  reported as a pilot, not counted. What it showed, and what this revision changes in response (fits that
  compute the same density about 10× faster; theories may no longer contain hand-set numbers; the incumbent
  is the simplest theory within one standard error of the best; theorists are told that short, general
  theories are the aim; a feasibility stage of one chain per arm), is listed in `PREREG_V2.md` §0 and its
  appendix. The hypotheses, arms, budget, participant rules and the bakeoff's plan are unchanged.
- **The method:** auto-psych runs automated discovery chains. In each experiment, coding agents (Gemini 4)
  propose computational theories of how people interpret or choose a speaker's message in simple reference
  games; every theory is fitted to all the chain's data with NUTS and compared by grouped cross-validation;
  the next experiment is designed to tell the surviving theories apart; real people on Prolific take it.
- **Two arms**, identical except for who chooses each experiment:
  - **open:** an agent designer proposes a new paradigm each experiment, which the harness checks, scores for
    expected information gain, pilots with simulated participants and freezes;
  - **fixed:** every experiment uses one frozen forced-choice paradigm (the earlier closed campaign's); only
    its displays are chosen, by the same information-gain tool.
- **The question** (Yarkoni's generalizability problem): do the open arm's final theories generalize across
  operationalizations better than the fixed arm's?
- **Stages:** (1) open chain 1 and fixed chain 1, live, together, judged by a registered feasibility gate;
  (2) on a pass, chains 2-4 of each arm; (3) a bakeoff on a blind paradigm, whose analysis is registered
  separately before any bakeoff data.
- **The record:** `docs/auto_open/PREREG_V2.md` and the code at the tag `rsa-open-prereg-v2` of
  https://github.com/mcfrank/auto-psych; decisions and reasons in `docs/auto_open/PLAN.md` and
  `docs/auto_open/REREGISTRATION.md`.

**Hypotheses:** (unchanged from the first registration)
- **H1 (transfer):** on a blind paradigm, the open arm's final theories predict new people better than the
  fixed arm's. On the fixed paradigm, the fixed arm's do at least as well.
- **H2 (parsimony):** at equal or better transfer, the open arm's finals use fewer theory parameters, and their
  parameters are more stable across paradigms. Both arms' theorists are now steered toward parsimony and
  both choose the incumbent by the one-SE rule, so parsimony is reported for the final incumbent and for the
  best theory by fit alone (which the rule does not shape).
- **H3 (the designer), descriptive:** agent-designed paradigms are valid and informative.
- H1 and H2 are tested in the bakeoff, under its own registration. **Confirmatory now: the feasibility gate**,
  a pass/fail decision on the stage-1 pair (Analysis plan).

### Design plan

**Study type:** Experiment: a sequence of online experiments, each designed from the data before it. The
between-chain comparison of the arms is the study's design.

**Blinding:**
- Participants do not know the hypotheses or the arm (one neutral study listing for both arms).
- The agents have no internet and never see held-out data, other chains' data or theories, or the earlier
  campaign's live or pre-registered data or final theories.
- The gate is a fixed script applied to the chains' run trees.

**Study design:**
- **Chains:** 4 per arm, each of **3 experiments**; chain k of each arm starts from the same seed theories, and
  chain 4 repeats chain 1's seeds. The literature data (50,587 trials) are every chain's first block.
- **Each experiment:** design → live collection → the inner loop (**3 rounds of 6 theorist agents**; admission
  gates, including **no hand-set numbers** in a theory; grouped cross-validation within each paradigm; a
  4-SE guard on every earlier paradigm; the incumbent chosen by the **one-SE rule**: of the eligible theories
  within 1 clustered SE of the best on the newest paradigm, the one with the fewest parameters; pruning) → the
  live set carried to the next experiment.
- **Bounds on a designed paradigm, the fixed paradigm and the budget** ($266.67 per experiment): as in the
  first registration (`PREREG_V2.md` §2-3).
- **How a chain runs:** one launch per chain, on the lab's own node; fits run in the chain's job.

**Randomization:** as in the first registration: balanced trial lists in arrival order; trial order and screen
positions permuted by a seed fixed at the design's freeze. The arms' chains run concurrently from stage 1 on.

### Sampling plan

**Existing data:** Registration prior to creation of data: no data of this registration's chains exist. The
pilot chain's data exist and are not part of any test (below).

**Explanation of existing data:** the literature data are every chain's first block and were used to build and
fit the seed theories. The pilot (open chain 1 of the first registration) collected live data for two
experiments and began a third; they are reported as the pilot, never pooled with this registration's chains
or used in the bakeoff. Looking at the pilot's theories and fit times motivated the changes this revision
registers (`PREREG_V2.md` §0). No fixed-arm chain has run live. The earlier closed campaign's data are not
used by either arm.

**Data collection procedures:** Prolific, desktop only, approval rate ≥ 98%; a person takes part at most once
in each chain; people may take part in other chains, of either arm, in the pilot and in the earlier campaign;
every study a person takes part in is recorded privately. Pay is each paradigm's own, at least $10 an hour
(the fixed arm: $0.50 for 3 minutes). Consent under the existing IRB protocol.

**Sample size:** per experiment, the open designer's N (≥ 100, within the budget) and the fixed arm's 400; 4
chains per arm × 3 experiments (24 experiments).

**Sample size rationale:** the per-experiment budget fixes the spend; the scale is set by measured running
times (`PREREG_V2.md` §3a). The unit of inference for the arms is the chain. Power for the bakeoff is set in
its registration.

**Stopping rule:** a study recruits until its N submissions or 24 hours, then is paused and its partial data
modelled. A chain stops after its third experiment, or at a designer failure after live data (reported as
shorter; never rerun); a designer failure before any live data restarts the chain once. If the gate fails, no
further chain runs under this registration.

### Variables

**Manipulated variables:** the arm (open or fixed), between chains.

**Measured variables:** each trial's response in the paradigm's format; comprehension checks, catch trials,
completion time, an end-of-study enjoyment rating (1-5; comments private).

**Indices:** each theory's out-of-fold log predictive density on each paradigm, and differences with
unit-clustered standard errors; the gate's indices; claim 3's and parsimony's measures (`PREREG_V2.md` §7).

### Analysis plan

**Statistical models:**
- **Fitting and selection** (every experiment, both arms): each theory and each paradigm's link fitted jointly
  by NUTS (4 chains × 1,000 + 1,000) to all the chain's data; eligibility within 4 clustered SEs of the best
  on every earlier paradigm; the incumbent by the one-SE rule; a fit limited to 4 hours; a theory whose fit or
  folds do not converge is not carried.
- **The feasibility gate** (confirmatory now), computed by `src/rsa_open/gate.py` on open chain 1 and fixed
  chain 1, experiments 1-3. It passes when, for **each** chain, all hold:
  1. each experiment froze a paradigm (open: within the designer's two cycles);
  2. in each, final comprehension ≥ 0.8, catch trials passed ≥ 0.8, exclusions ≤ 25%, the data parse, and
     ≤ 10% of readable participants give the same answer on ≥ 90% of test trials;
  3. in each, the loop's incumbent at its first scoring beats a chance theory on the new paradigm by more
     than 4 unit-clustered SEs;
  4. each loop ends with ≥ 2 trusted theories and an export, and ≥ 1 new theory is admitted over the 3
     experiments;
  5. each experiment runs within 48 hours (design to export, recruitment excluded).
  Agent spend, total running time and participant payments are reported, not judged.
- **The bakeoff:** fixed by its own registration (`PREREG_V2.md` §6).

**Transformations:** none. Catch and practice trials are not modelled.

**Inference criteria:** the gate passes only if every criterion holds for both chains (one decision, no
correction). On a pass, the pair counts as chain 1 of each arm and chains 2-4 of both arms run. On a failure,
no further chain runs; a revision is registered anew and the pair is reported as a pilot.

**Data exclusion:** as in the first registration: missed catch trials, failed comprehension on the second
reading, an earlier experiment of the same chain, or an incomplete record. Nothing else.

**Missing data:** only submitted, complete sessions are analysed; an underfilled study's partial data are
modelled.

**Exploratory analysis** (reported, not tested): claim 3's measures; admission outcomes; parsimony per theory
and arm (parameters, lines, the incumbent and the best by fit alone, how often the one-SE rule changed the
incumbent, candidates refused for hand-set numbers); the prospective test (each committed theory scored on the
next experiment's people, against its cross-validated score); returning participants and the first-study
sensitivity analysis; the pilot and the rehearsals.

### Other

**Deviations:** any deviation is reported with its reason, in a dated addendum to `PREREG_V2.md` and in the
paper.

**AI disclosure:** The text of this registration (`PREREG_V2.md` and `OSF_PREREG_V2.md`) was drafted by Claude
Opus 5.5 (Anthropic), an AI model, working with Michael C. Frank in Claude Code, which also wrote much of the
campaign's code (its commits name it as co-author). Michael C. Frank made every design decision recorded here,
reviewed and approved the text, and is responsible for it. Separately, AI agents are part of the method under
study: the designer, theorists and critic are Gemini 4 models, and the pilot's simulated participants Gemini
3.8 Flash.

**The frozen materials:** the git tag `rsa-open-prereg-v2` of https://github.com/mcfrank/auto-psych, commit
`b32e732b74d99d5f3dd85a9374d7d2e6b44beab8`.

| file (in that commit) | sha256 |
|---|---|
| `src/rsa_open/gate.py` (the gate) | `0ae779367bdd513a8945704dd90b7b2033852b7cb6f5debf801a11a814c7a6e1` |
| `src/rsa_open/fixed_paradigm.json` (the fixed arm's paradigm) | `98247195c5755fcee16ebfe7f65f04c599f224c8774b3c8a6e52c004ba0cdda1` |
| `src/rsa_open/loop/orchestrator.py` (the loop; the chance theory) | `7750355fab1cda82a7f8f41b581ed6547b19fa4e4c1e7ae42be0679c6275dcc8` |
| `src/rsa_open/corpus.py` (the guard and the one-SE rule) | `fbd566db819246333c1c3074a2d9e5ae5cfcbe9fe1cc884debf0ba8f36bb534c` |
| `src/rsa_open/loop/gates.py` (admission, no hand-set numbers) | `511081513fa1c9637d6446035c63d293834f6d5e64f415e3070ea678ca149af4` |
| `src/rsa_open/loop/prompts/open_theory.md` (the theorists' brief) | `6687696fab1c37724b5ebbb78c12a3b23ad3e83e394d0b18edc122465061a13d` |

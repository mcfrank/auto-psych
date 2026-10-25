# The open campaign's final bakeoff: draft pre-registration

**Status: draft, not registered.** It is finalized and registered on OSF after the open campaign's
chains finish and before any bakeoff participant is recruited, as `rsa-prereg-final` was for the
closed campaign (`docs/auto_rsa/PREREG_FINAL.md`). Sections marked **TBD** are fixed at
registration. The plan it implements is `PLAN.md` §1 (claims) and §7 (the bakeoff).

## 1. Claims under test

1. **Transfer.** On a blind paradigm, the open campaign's final theories predict new people better
   than the closed campaign's finals (and than the seeds).
2. **Parsimony.** At equal or better transfer, they use fewer theory parameters, and their
   parameters are more stable across paradigms (each paradigm fitted alone, estimates compared).
3. **The designer.** Agent-designed paradigms are valid and informative: realized information gain
   against predicted EIG, comprehension and catch pass rates, pilot against live.

## 2. Competitors

- **Open finals:** each of the open campaign's three chains' final incumbent.
- **Closed finals:** the closed campaign's three finals, ported to theory files (`PLAN.md` §3.1).
- **Upgraded closed finals (supplemental, §5):** if that arm is run.
- **The seeds**, as a baseline.

## 3. Arms

- **F, forced choice:** the closed campaign's paradigm on fresh displays, designed with its own
  rule. This is the closed theories' home ground.
- **T, transfer:** a blind paradigm with a new response format, designed by an isolated designer
  run that no chain or theorist has seen.

## 4. Fits and the primary comparison

- **A (primary):** every competitor fitted on the same pooled data: E, plus both campaigns' live
  data, never the closed campaign's pre-registered data. This compares theories.
- **B (secondary):** each competitor fitted on its own campaign's data. This compares methods,
  including the data each campaign chose to collect.
- **Predictions:** the open finals beat the closed finals in T. F is tested the same way, and
  there the closed finals are expected to do at least as well.
- **TBD:** the score (held-out log predictive density per participant, as the closed prereg),
  the test and its threshold, power and sample sizes, exclusions, and the frozen pool predictions
  with their hashes.

## 5. Supplemental comparison: the upgraded closed arm (PI 2026-10-25)

**Why.** By the bakeoff, the open campaign's theorists will have worked with tooling the closed
campaign's theorists never had (`PLAN.md`, Decisions 2026-10-23 and 2026-10-24):
- a self-check that reports the sampler's steps per draw, step size and correlated parameter
  pairs, and projects the admission fit's time;
- rejection reasons that carry the same report;
- a prompt that names RSA's identification pairs (rationality and cost, rationality and listener
  sharpness, prior weights and lapse);
- an admission limit relative to the corpus;
- near-miss-only refits and bounded cross-validation folds.

The identification prompt bears directly on claim 2, and the tooling as a whole may help claim 1.
Comparing open against closed alone would confound the designer and the changing paradigm with
better tooling for the theorists.

**What.** The closed campaign's **nine inner loops** (3 chains × 3 experiments) rerun in order on
that campaign's own frozen designs and collected live data, with the open campaign's theorist
tooling and harness rules (the list above). Everything else stays as in the closed campaign:
- the theorist model, the sandbox, the rounds and slots;
- the seeds and chain split;
- the closed loop's selection rule.

Each experiment's loop starts from this arm's own previous carried set, not the original closed
campaign's. Each chain's final incumbent is an **upgraded closed final**. These models may use the
closed campaign's live data, as the closed finals do, but never its pre-registered data.

**When.** The arm is run only if the open campaign completes its chains and proceeds to the
bakeoff. That decision is made when the campaign ends, before any bakeoff data exist, and does not
depend on any bakeoff outcome. If the open campaign is abandoned, the arm is not run. When run, its
finals and their pool predictions are frozen and hashed with the other competitors' before
recruitment.

**Analysis.** In both arms (F, T) and both fits (A, B):
- open finals against upgraded closed finals attributes a difference to the designer and the
  changing paradigms, with the theorists' tooling equal;
- upgraded closed finals against closed finals attributes a difference to the tooling.

Claim 2's parameter counts and cross-paradigm stability are reported for all three sets.

**Known confound.** The closed designs were chosen by the original closed loops' carried sets, not
by this arm's, so the upgraded closed theorists work on data selected for different theories.
This is stated, not corrected.

## 6. Also reported (descriptive, not tested)

- Each campaign's admission outcomes: too slow, not converged, admitted. The closed campaign's
  30-minute limit and the open campaign's relative limit are reported side by side.
- Claim 3's designer measures, pilot against live.
- The rehearsals' records (`RUN_B_REPORT.md`, the replay of run B's experiment 1).

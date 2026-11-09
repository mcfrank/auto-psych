# Re-registering the open campaign (working list, 2026-11-08)

Written by Claude Opus 5.5 for the PI; not reviewed by a person. A running list of what the new
registration has to cover, so nothing decided since the first one (https://osf.io/78t69, tags
`rsa-open-prereg`, `-a1`, `-a2`) is left out. Decisions are the PI's; open ones are marked.

## Decided

- **Chain 1 (`chain_0` in the code) is a pilot** (PI 11-08), as §4 of the registration allows for a
  failed gate. Its data, theories and timings are reported as the pilot, not counted in the bakeoff.
  Reasons: it ran with per-trial fitting (slow fits shaped some admission and trust decisions), and
  its theories carry dozens of hand-set constants each, which the new gate refuses.
- **No hand-set numbers** (PI 11-08, strict): a theory's numbers outside `PARAMS` are 0, 1, memo_kit's
  `EPS` and `BIG`, and integers that lay out arrays (`gates.magic_number_problems`, in admission and
  the self-check; the theorists' brief says why). Chain 1's winners had 15 and 85 tuned decimals with
  2 free parameters; the closed campaign's finals 0-3 with 9-17. Seeds were made to pass (named
  constants, identical predictions).
- **Fit speed, no change to what is computed:** each distinct display evaluated once and each pattern
  once (786e609b); one compiled function per block's likelihood (9664699f); NUTS reaches the
  density through a callback compiled once (d2fcfb25); the theory evaluated once per display shape
  across blocks, not per block.

- **The one-SE rule** (PI 11-08), both arms: of the trusted, eligible theories within 1 clustered SE
  of the best on the newest paradigm, the one with the fewest parameters leads
  (`corpus.simplest_adequate`, `parsimony_k`); pruning still measures distance from the best fit.
  The theorists are told the aim (short theories that carry over), the rule, and that simplifying
  is progress (`open_theory.md`; a refinement may be a simplification). Parsimony stays an outcome,
  measured twice: the incumbent's (which the rule shapes) and the best-by-fit-alone theory's
  (`fit_best` in `history.json`, which the rule does not touch); the registration says both arms'
  theorists were steered toward parsimony.
- **Fits in the step's own job** (PI 11-09, after the stepped rehearsal, review V3 §16: 316 of 325 fit
  tasks under rule 10's 10-minute line): `STEP_FITS=local`, each step runs its fits on its own cores
  and the chain continues in the same job (`HANDOFF_centralized_fits.md`). A launch submits no job but
  a continuation when its time runs short. `LAUNCH_EXPERIMENTS=all` lets a launch run the whole chain
  (open: whether to use it, below).
- **The feasibility gate's wall time: 48 h** (PI 11-09). Proposed reading: each experiment, launch to
  export, excluding recruitment and waits for a person, within 48 h (the stepped rehearsal projects at
  most about 21 h for the open arm's experiment 3). To confirm: per experiment, not for the chain
  (the open chain projects 48-53 h for three experiments).
- **The plan to registration** (PI 11-08):
  1. Revise the theorist prompt for the hand-set-numbers behaviour (parsimony), before the next
     rehearsal.
  2. A stepped rehearsal (`HANDOFF_stepped_rehearsal.md`): both arms, one experiment each, 3 rounds
     of 6, real agents and simulated participants (no Prolific). It measures each step's wall time, queue waits and
     failures at the new fit speeds; if the fit arrays mostly wait in the queue, the step job runs
     its own fits instead.
  3. Register, with these measurements, a stage 1 of one open and one fixed ("closed") chain run
     together for 3 experiments, behind feasibility gates (the PI calls this pair the pilot), then
     3 more chains per arm.
- **Gates are on feasibility only** (proposed, 11-08): wall time for 3 experiments, convergence,
  no code change mid-chain, cost per experiment. Nothing that compares the arms, so the
  confirmatory comparison stays clean. The stage-1 pair counts toward the 4 per arm if the gate
  passes with no protocol change, and becomes a pilot otherwise (to be fixed in the registration).
- **Claims as estimates:** with 4 chains per arm, outcomes (transfer, prospective prediction,
  parsimony) are reported as estimates, not a significance test. The arms run side by side, with
  alternating launch order, so recruitment time is not confounded with arm.

## Open

- **Autonomous chains** (`LAUNCH_EXPERIMENTS=all`): SRCC heard on 11-05 that a person starts one job per
  experiment; with fits in the step a whole chain can be one job (up to 7 days on `mcfrank`) that
  submits nothing. Tell SRCC before using it. For live chains, each experiment publishes a Prolific
  study, so the launch approves every experiment's spend.

- **Scale and scope:** rounds per experiment (PI leaning 3), experiments per chain (3 or 4), chains
  per arm, candidates per round, repairs, the carry cap; set against the fit times measured after
  the speed work, not chain 1's.
- **Fit settings:** full 1,000 + 1,000 or cheaper (the refit test, `HANDOFF_refit_test.md`), B1 folds.
- **Admission limit:** its factor and an absolute cap.
- **Architecture:** one job per experiment with fits inside it, or step mode, decided by real fit
  times (`HANDOFF_fit_speed.md`, review V3 §13). Either way an infrastructure section.
- **Gate items:** what the feasibility gate becomes when chain 1 is a pilot (a new feasibility chain,
  or the gate dropped in favour of the pilot's evidence).
- **Parsimony measures** to report per arm: free parameters, attempted hand-set constants (refused
  by the gate), structural size, and the gain over each theory's own core recursion.
- **Prospective test:** each committed theory scored on the next experiment's data, which no agent
  saw (the closed campaign's claim 2), as a check on cross-validation.
- **Remaining compile cost:** NUTS's sampling loop takes about 90 s and several GB to compile for a
  large theory (review of 11-08 in this session's notes); a persistent compile cache shared by a
  theory's fit and its folds, or compiling once per theory, would remove most of it.

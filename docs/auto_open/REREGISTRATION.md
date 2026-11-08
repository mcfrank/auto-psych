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

## Open

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

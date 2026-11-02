---
name: session-drift-response-stage
description: experiment2 trial_index drift is weak; drifting lapse guess-rate +2.8 log-post at MAP, other forms worse
metadata:
  type: project
---
Tried 2026-10-30 (experiment2 iter_0 candidate_5, `drifting_default_answer_concepts`, on the final_digit_multiplicative base). MAP log-posterior on experiment2: base -2230.7, incumbent costly_squares -2231.6 (the squares split doesn't help here).
- Lapse with a guess rate that is logit-linear in trial_index/450: -2227.9 (lapse 0.013, guess logit -0.42 at start and -1.42 at end).
- A disengaged "no" rising over the session: -2231.6. A log-odds criterion shift drifting over the session: -2230.1.
- MAP residuals by p bin x session fifth are small (|resid| < 0.07). Mid-p (0.2-0.8) targets get +0.06 yes early and slightly fewer late.

**Why:** most of the yes-rate fall (0.52 to 0.25) comes from the stimulus order, so the critique's z=-2.5 leaves little to gain.
**How to apply:** don't expect big ELPD gains from session drift. To read MAP p_yes, use mp["p_yes"] from find_MAP. Related: [[experiment2-data-quirks]].

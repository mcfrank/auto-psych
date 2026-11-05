---
name: guessing-lapse-ceiling
description: lapse+guess-rate on the exp2 incumbent gives +5 MAP; fits as a yes-ceiling (lapse 0.25, guess_yes 0.07); trial_index is banned in models
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_1/candidate_1_repair_1, inattentive_guessing_lapse). The incumbent seed_overshadowing_last_digit_recency has MAP -3152.7. Adding p = (1-lapse)*sigmoid + lapse*guess gives -3147.7, with lapse about 0.25 and guess_yes about 0.07, so the maximum p_yes is about 0.77. Smoke log-lik -3154.9; the check takes about 2 min.
The first attempt (a session-relaxing criterion using trial_index) was rejected: models may not bind trial_index, because the novelty and held-out pools contain only bare stimuli. So the early_session_yes_gap critique cannot be modelled.
**Why:** this is a decision-rule slot. The gain is small but it does not depend on set size, so it is distinct from [[single-example-leniency]].
**How to apply:** never use trial_index. A lapse is mostly a ceiling on yes-rates.

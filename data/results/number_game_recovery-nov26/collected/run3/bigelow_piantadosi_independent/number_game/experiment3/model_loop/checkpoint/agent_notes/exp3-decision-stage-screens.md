---
name: exp3-decision-stage-screens
description: exp3 decision-rule screens on repdigit incumbent — all near null; singleton-only guessing +2.4 loglik best
metadata:
  type: project
---
2026-10-31, exp3 iter_0/candidate_0 (`single_example_guessing`). find_MAP on repdigit_pattern_chart_concepts (base loglik -3465.9, the incumbent's lapse goes to 0). Results:
- uncertainty-gated guessing (both the 0.5 and the fitted-guess versions): giveup goes to 0, gain 0.0. This was +23 in exp2 ([[uncertainty-triggered-guessing]]) but does not replicate.
- logit yes-bias: +0.4. Gain plus bias: +0.7 (gain 1.04).
- separate coin-flip guess share for single-example sets: +2.4 (share 0.15). This one was submitted; its smoke loglik was -3470.3 and the smoke fit took 95 s.
The machine has 1 CPU, so each MAP screen takes ~3-4 min.

**Why:** on exp3 the residuals are not in the response stage.
**How to apply:** look for exp3 gains in the concepts and the prior, not in the lapse or bias form.

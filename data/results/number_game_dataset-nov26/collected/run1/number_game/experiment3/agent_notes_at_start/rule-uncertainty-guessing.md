---
name: rule-uncertainty-guessing
description: exp2 iter1 c1 — plain exemplar + guessing that falls as the set's rule gets clearer; MAP -57 nats vs incumbent
metadata:
  type: project
---
Exp2 iter_1 candidate_1 (2026-11-02), a decision-rule slot aimed at the background_set_rate_sd critique. Model: fixed-weight similarity on magnitude, digit, parity and set divisor. The guess weight is sigmoid(base - drop*clarity) with a free yes-rate. MAP log-lik -2771.5 vs incumbent -2714.4. Smoke fit -2774.6. At MAP, singletons are about 75% guesses.
Grafting the same guessing onto the incumbent gave MAP -2727, no gain: coincidence sharpening already covers this.
**Why:** shows that coherence-gated guessing does not explain why sets differ in breadth beyond what sharpening does.
**How to apply:** don't re-try decision-level guessing on the incumbent; see [[exp2-iter1-critique-stats]].

---
name: all-or-none-rule-commitment-fails
description: Committing to the single smallest consistent rule (no posterior averaging) fits ~170 nats worse than the best Bayesian model
metadata:
  type: project
---
Round 1 candidate 2 (2026-10-30), `smallest_rule_commitment`: commit to the smallest consistent rule out of 44 arithmetic+digit rules (span fallback), plus 2 yes-rates. Smoke-fit log-lik was about -1088, against an ELPD of -918 for `tenenbaum_digit_concepts`. Only 200 of 2250 trials needed the fallback.

**Why:** people's generalization is graded, so averaging over hypotheses (and intervals) matters. Hard MAP commitment is too coarse.
**How to apply:** don't bother with more variants of deterministic MAP/rule commitment. Graded posterior mixing is needed. See [[digit-concepts-key]].

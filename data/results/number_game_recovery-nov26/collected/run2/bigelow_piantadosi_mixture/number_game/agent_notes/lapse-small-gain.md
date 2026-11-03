---
name: lapse-small-gain
description: Adding guess-lapse or free noise to tenenbaum_digit_concepts gains only ~2-4 nats smoke loglik; a hand-rolled find_MAP loglik script gave bogus numbers
metadata:
  type: project
---
Round 1 candidate_4 (2026-10-30): `digit_concepts_with_guess_lapse` (incumbent + eps/g lapse) smoke loglik -910.4; free NOISE ~Beta(2,8) variant -912.2; incumbent ~-914.4 ([[digit-concepts-key]]).

**Why:** my own MAP script, built with pm.logp + compile_fn, reported -1264 for the base model, which is wrong. Trust check_candidate's smoke loglik instead.
**How to apply:** small response-level tweaks are close to saturated; bigger gains probably need hypothesis-space changes.

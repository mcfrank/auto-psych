---
name: set-kin-consensus-exemplar
description: exp3 iter3 c5: routes × exp(-rho·share of examples target isn't kin to) on parity_preserving_decisive_exemplar, +19 MAP; smoke -4057.3
metadata:
  type: project
---
exp3 iter_3 candidate_5 `set_kin_consensus_exemplar` (2026-11-02). It takes parity_preserving_decisive_exemplar and multiplies every distance-free route (multiple, factor-kin, round) by exp(-rho·kin_unshared). kin_unshared = 1 − (fraction of examples the target is multiple/gcd>=3/round kin to). MAP -4048.8 vs base -4067.8 and incumbent route -4066.0. rho≈1.9. Smoke -4057.3.
Diagnosis: far relatives of one example in incoherent multi-example sets are over-predicted. Multiples are 0.17 observed vs 0.47 predicted, digit matches 0.22 vs 0.40. Single-example far relatives fit well. The polarization stats barely move (0.12-0.15 predicted).
Screened with no gain: additive digit route gated by digit homogeneity (0); replacing the multiplicative digit penalty with a route (-53); n^-rho route dilution (+6).
**Why:** follows [[exp3-iter3-critique-stats]], [[parity-preserving-decisive-exemplar]].
**How to apply:** far digit-match over-endorsement in incoherent sets is still open. Screening scripts are in /tmp/c5s (fit.py, rel.py). Note that compile_fn needs replace_rvs_by_values.

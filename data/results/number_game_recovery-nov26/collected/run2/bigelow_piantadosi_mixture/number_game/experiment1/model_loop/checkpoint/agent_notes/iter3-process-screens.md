---
name: iter3-process-screens
description: Round 3 screens of new process families (memory, carry-over, log number line, pedagogical) all tie the base; Chebyshev-in-parameter trick makes heavy models ~0.3 ms/grad
metadata:
  type: project
---
Round 3, candidate_2 (2026-10-30), MAP screens on equal_prior+lapse (NLL 904.52):
- Random forgetting of examples: forget rate goes to 0, no gain.
- Carry-over of the previous trial's set or target into the current one: c goes to 0. Trials are interleaved, and there are only 450 distinct (set, target) cells × 5 sessions.
- Power/log compressed number line for intervals: log is worse (+19 nats); the best exponent is about 1.13, a gain of 0.3.
- Pedagogical teacher, P(x|h) ∝ P(h|x)^α: α≈0.05-0.2, gain at most 0.1. I submitted it as `pedagogical_sampling_concepts` (smoke loglik -906.4; RMSE vs base about 0.005 at α=0.19).

Speed trick: when the only nonlinear parameter enters the hypothesis likelihoods, tabulate per-set and per-cell log-summed rule and interval likelihoods at Chebyshev nodes in `prepare_observed` (degree 16, error about 1e-10). λ then enters exactly via logaddexp. This took logp+grad from 171 ms to 0.3 ms.

**Why:** this confirms [[number-game-ceiling]]. New process families tie at best.
**How to apply:** don't re-screen these. Use the Chebyshev tabulation for any one-parameter likelihood change (see [[fit-speed-budget]]).

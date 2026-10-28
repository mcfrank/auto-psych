---
name: exp3-r3-residual-scan
description: Exp3 round 3 MAP residual scan of upward_count_start_boundary — which single additions gain (even rule ~9 nats) and which do nothing
metadata:
  type: project
---
Exp3 round 3 (2026-10-18), MAP log-lik of the incumbent upward_count_start_boundary = -4642.8. Single additions tested:
- "even numbers" rule for all-even sets, (k-1)·log2·±1, own HalfNormal weight: -4634.1 (+8.6). A matching odd rule fits ~0, so the effect is asymmetric. Submitted as even_number_rule_abstraction (iter_3/candidate_3).
- Even-target main bias +6; extending the factor rule to g>=2 +4; g-exponent on the factor rule +5; set-size criterion +3.5; set-size breadth +1; partial span filling 0.
- Normalizing the factor mismatch by log100 or by log max instead of log min: about 30 nats WORSE. Keep min-normalization.
Leftover residuals: the factor rule over-predicts for g=3 and g=9 sets and under-predicts for g=2/4/5/8. Targets 1 and 2 are over-predicted by ~0.07. Set-level SD is still compressed (0.147 observed vs 0.116 predicted).
**Why:** saves re-running dead ends next round.
**How to apply:** start from scratch scripts in iter_3/candidate_3/scratch (resid.py, mapvals.py, grp*.py). See [[residual-ascending-below-start]].

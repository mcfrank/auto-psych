---
name: exp3-iter2-sequential-kind-probes
description: Exp3 iter_2 cand_0 (2026-11-01) — learned example reliability pins at 0.9; first-example kind commitment weight ~1.03 (+0.4 nats, RMSE 0.008), submitted as sequential_kind_revision
metadata:
  type: project
---
Fast base (no centre weighting, no graded dsum, equal-share digit sums, Erlang ranges): MAP data LL ≈ -3515.0 (incumbent -3513.5).
- Learned reliability ALPHA: profile optimum ≈0.905, so 0.9 is already right. Dead.
- Kind posterior ∝ prior × first-example kind evidence × (later evidence ratio)^w, with full Bayes within each kind: w=1 -3515.0, 1.05 -3514.6, 1.1 -3520, 0.9 -3530, 0.5 -3864. MAP w 1.03. Moving w from 1 to 1.03 shifts p_yes by RMSE 0.008, so it should pass novelty. Sampling 200 iterations takes 6 s. Submitted as sequential_kind_revision with a LogNormal(0, 0.5) prior on w.
**How to apply:** kind-level evidence weighting is identified but only just interior; don't re-probe. See [[exp3-iter1-process-probes]].

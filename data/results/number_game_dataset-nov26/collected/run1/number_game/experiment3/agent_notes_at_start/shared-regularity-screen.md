---
name: shared-regularity-screen
description: iter_3 cand_0 — set-level "most specific shared rule" logit term on graded-parity incumbent gives ~7 nats in-sample; per-example category features ~0
metadata:
  type: project
---
iter_3 candidate_0 (2026-11-02), on graded_parity_variability_exemplar (MLE NLL 1561.3): adding rule_weight*(surprise of most specific shared concept the target satisfies − that of most specific one it violates), with concepts mult3-12, powers, squares, cubes, primes, odd/even, end-digit, surprise log(100/size), sets of n>=2 only → NLL 1554.5. Penalty only or bonus only: 1-3 nats. Per-example category-feature similarity dims (Tversky-style): <1.5 nats. Decade concepts add nothing. Submitted as `shared_regularity_exemplar` (smoke log-lik -1559.2).
**Why:** biggest misfits are rule-coherent sets (3,63 → 36,39 yes; 8,32 → 37 no; 78,48,88 near non-matches no), but the gain from rule evidence is modest.
**How to apply:** don't re-screen per-example arithmetic-category similarity; see [[iter3-critique-stats]].

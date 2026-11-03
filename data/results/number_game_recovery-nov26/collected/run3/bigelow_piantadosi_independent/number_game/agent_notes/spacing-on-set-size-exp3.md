---
name: spacing-on-set-size-exp3
description: exp3 iter2 cand5 — spacing shift on set_size_evidence_privileged_fives +4.7 MAP; tens-digit decade rows null
metadata:
  type: project
---
2026-10-31, exp3 iter_2/candidate_5 (`spacing_sparsity_graded_evidence`), a refinement slot. Base incumbent set_size_evidence_privileged_fives has MAP loglik -3423.2 (21 s, screen script /tmp/c5i2/screen.py).
- Adding the mean-gap logit shift (coef -0.53) gives -3418.4, +4.7. This was submitted as the spacing model plus a set-size graft.
- Check OK, smoke loglik -3426.4.
- Tens-digit decade rows (30-39) instead of 31-40: -3423.4, null. Bonusing both row types: +1.8.
- The iter2 critique's only hit was round targets after non-round sets (-0.057 vs 0.004, q=0.064). Row redefinitions do not close it.
**Why:** single-parameter changes on the exp3 incumbent keep landing at ≤5 nats.
**How to apply:** if an incumbent slot also adds spacing, this candidate is a duplicate. Related: [[exp3-spacing-sparsity-and-nulls]], [[exp3-iter2-critique-stats]].

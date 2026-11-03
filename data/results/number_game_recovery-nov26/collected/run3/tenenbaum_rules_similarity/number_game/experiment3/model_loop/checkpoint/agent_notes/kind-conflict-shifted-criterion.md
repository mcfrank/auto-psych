---
name: kind-conflict-shifted-criterion
description: exp3 iter_2 c5: per-target between-kind disagreement shifts criterion (refines uncertainty_shifted_criterion), smoke ll -3654.2
metadata:
  type: project
---
exp3 iter_2 candidate_5 (2026-10-31) is `kind_conflict_shifted_criterion`. It is built on the incumbent's code. The criterion shift is −c·4·Σ_k w_k(p_k − p)², where w_k is the kind posterior (7 rule families + range) and p_k is the kind-conditional membership of the target. FAMILY_MAT (rules×families) does the matmul. Unlike the set-level entropy, this shift is target-specific. Smoke ll is −3654.2 (≈ the others). I did not check c's posterior or novelty.
**Why:** if c≈0, this model duplicates beta_share at the gate.
**How to apply:** if it gets rejected as a dup, the per-target conflict idea is null too. See [[kind-uncertainty-shifted-criterion]], [[rule-vs-range-doubt-criterion]].

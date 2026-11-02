---
name: specificity-biased-rule-prior
description: exp2 iter_2 cand_2 — within-family rule prior ∝ size^(-specificity) on beta-share incumbent; specificity≈0.10±0.11, lppd −2263.9 (2×300 NUTS)
metadata:
  type: project
---
Motivated by the iter_2 critique nested_multiple_specificity (obs 0.49 > null 0.39, q=0.28).
Grafted onto beta_share_width_exception_concepts: within each family, rule prior ∝ (members)^(-specificity), normalised per family, Normal(0,1) prior.
Quick 2×300 NUTS (2026-10-30): specificity 0.10±0.11 (≈0), in-sample lppd −2263.9; smoke fit 11 s. Likely a near-null / possible novelty-duplicate.
Raw data: 8,80,48 → multiples of 8 5/5, other evens ≈0; but 2-example sets (6,66; 8,32) have only weak yes on multiples.
**Why:** another within-concept-prior tweak that exp2 can't distinguish; see [[experiment2-null-refinements]].
**How to apply:** don't retry rule-size prior exponents on exp2.

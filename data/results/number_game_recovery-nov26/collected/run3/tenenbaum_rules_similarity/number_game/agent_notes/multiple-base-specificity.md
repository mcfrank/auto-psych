---
name: multiple-base-specificity
description: exp2 iter_2 cand5: multiples-of-m prior ∝ m^exponent on beta_share_parity_range — exponent ≈ -0.01±0.14, null
metadata:
  type: project
---
experiment2 iter_2/candidate_5 (2026-10-30), multiple_base_specificity_concepts: aimed at critique nested_multiple_specificity (obs 0.495 vs null 0.388, q=0.28). Base exponent posterior -0.01 ± 0.14 (1 chain, 200 draws); smoke ll -2270.1. So the within-family prior over multiples doesn't close the gap; likely a near-duplicate of [[beta-share-parity-range]].
**Why:** rule-prior tweaks keep collapsing at exp2 ([[concept-space-additions-map]]).
**How to apply:** don't retry base-dependent multiples priors; the nested-multiple gap needs a different mechanism (or is noise, q>0.05).

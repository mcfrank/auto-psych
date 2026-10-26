# Critique of `contextual_contrast_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prior_empty_vs_nonempty_singleton_unit22** — Points allocated to empty singleton [0, 0, 0] minus points allocated to 2-feature singleton [0, 1, 1] on uninformative prior trials with duplicate full distractors. (observed 2.12 vs null mean 0.015, z=3.30, p=0.002, q=0.016) [survives FDR]
- **multiclone_competitor_margin_unit20** — Points allocated to singleton competitor [1, 1, 1] minus mean points allocated to three identical clones [0, 0, 1] when hearing feature 2. (observed 0.3 vs null mean -0.772, z=2.89, p=0.00599, q=0.024) [survives FDR]
- **prior_dense_over_sparse_no_empty_scenes** — Mean points allocated to full 3-feature objects [1, 1, 1] minus mean points allocated to minimal 1-feature objects [0, 0, 1] on uninformative prior trials without empty objects. (observed 0.8 vs null mean -0.0017, z=2.39, p=0.02, q=0.0384) [survives FDR]
- **prior_empty_over_competitors_distinct_unit19_unit36** — Points allocated to empty object [0, 0, 0] minus mean points allocated to all other objects across 4-object displays where all objects are distinct. (observed 0.667 vs null mean 0.0145, z=2.34, p=0.024, q=0.0384) [survives FDR]
- **scalar_minimal_target_margin_contexts** — Mean points on minimal target [0, 0, 1] minus mean points on competitor objects across scalar implicature scenes with duplicate distractors when hearing feature 2. (observed 7.62 vs null mean 7.01, z=2.26, p=0.024, q=0.0384) [survives FDR]
- **prior_empty_duplicates_allocation_unit33** — Total points allocated to duplicate empty objects [0, 0, 0] minus points allocated to 3-feature singleton [1, 1, 1] on uninformative prior trials. (observed -1.57 vs null mean -2.6, z=2.04, p=0.032, q=0.0426) [survives FDR]
- **prior_dense_vs_sparse_distinct_unit42** — Points allocated to dense 3-feature object [1, 1, 1] minus points allocated to sparse 1-feature object [0, 0, 1] on uninformative prior trials where all objects are distinct. (observed 1.1 vs null mean 0.0119, z=1.95, p=0.042, q=0.046) [survives FDR]
- **duplicate_competitor_suppression_unit24** — Total points allocated to duplicate competitors [1, 1, 1] when hearing feature 2 in scenes with a minimal target [0, 0, 1] and an empty distractor [0, 0, 0]. (observed 0.525 vs null mean 0.906, z=-1.90, p=0.046, q=0.046) [survives FDR]

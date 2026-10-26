# Critique of `rsa_l2_singleton_feat_color_valence_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **scalar_competitor_points_unit2** — Mean points allocated to the overspecified competitor [1, 1, 1] when hearing feature 2 in the 4-object scalar scene with duplicate foils [1, 1, 0]. (observed 3.55 vs null mean 1.49, z=7.92, p=0.002, q=0.0032) [survives FDR]
- **universal_feature_simplicity_advantage_unit35** — Mean points on minimal 1-feature objects [0, 0, 1] minus mean points on extra-feature objects [0, 1, 1] and [1, 0, 1] when heard feature 2 applies to all objects. (observed 2.08 vs null mean -0.128, z=6.22, p=0.002, q=0.0032) [survives FDR]
- **singleton_competitor_inversion_unit25** — Points allocated to the singleton competitor [0, 1, 1] minus mean points allocated to duplicate minimal targets [0, 0, 1] when hearing feature 2. (observed -1.15 vs null mean 1.65, z=-5.20, p=0.002, q=0.0032) [survives FDR]
- **duplicate_targets_vs_singleton_competitor_unit27** — Total points on duplicate minimal targets [1, 1, 0] minus points on singleton competitor [1, 1, 1] when hearing feature 0. (observed 1.52 vs null mean -1.36, z=3.85, p=0.002, q=0.0032) [survives FDR]
- **prior_sparse_vs_dense_singleton_unit13** — Points allocated to 1-feature singleton [0, 0, 1] minus points allocated to 3-feature singleton [1, 1, 1] on uninformative prior trials with duplicate empty objects. (observed 0.725 vs null mean -1.64, z=3.60, p=0.002, q=0.0032) [survives FDR]
- **multiclone_competitor_allocation_unit20** — Mean points allocated to the singleton competitor [1, 1, 1] in contexts with three identical clones of minimal target [0, 0, 1] when hearing feature 2. (observed 2.73 vs null mean 1.81, z=3.30, p=0.00799, q=0.00913) [survives FDR]
- **prior_empty_over_full_distinct_unit19** — Points allocated to empty object [0, 0, 0] minus points allocated to full object [1, 1, 1] on prior trials where all four objects are distinct. (observed 0.925 vs null mean -0.691, z=3.05, p=0.004, q=0.00533) [survives FDR]
- **prior_empty_singleton_salience_unit29** — Mean points allocated to the empty singleton object [0, 0, 0] on prior trials with duplicate 1-feature objects and a 2-feature object. (observed 4.12 vs null mean 3.12, z=2.96, p=0.00999, q=0.00999) [survives FDR]

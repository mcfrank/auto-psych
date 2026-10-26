# Critique of `contextual_contrast_listener`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prior_empty_vs_nonempty_singleton_unit22** — Points allocated to empty singleton [0, 0, 0] minus points allocated to 2-feature singleton [0, 1, 1] on uninformative prior trials with duplicate full distractors. (observed 2.12 vs null mean 0.0205, z=3.50, p=0.004, q=0.032) [survives FDR]
- **multiclone_competitor_margin_unit20** — Points allocated to singleton competitor [1, 1, 1] minus mean points allocated to three identical clones [0, 0, 1] when hearing feature 2. (observed 0.3 vs null mean -0.75, z=2.93, p=0.00999, q=0.04) [survives FDR]
- **prior_dense_vs_sparse_distinct_unit42** — Points allocated to dense 3-feature object [1, 1, 1] minus points allocated to sparse 1-feature object [0, 0, 1] on uninformative prior trials where all objects are distinct. (observed 1.1 vs null mean -0.0234, z=2.10, p=0.048, q=0.108)

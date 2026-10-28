# Critique of `zigzag_breadth_sequence_exemplar`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **small_vs_large_region_gap** — Among targets outside the span by a linear gap of 5-40, yes-rate when the nearest example is <=20 minus when the nearest example is >=60; observed < null means people compress large numbers (log-like magnitude) so equal linear gaps matter more among small numbers, which the linear-distance model fails to produce. (observed -0.0584 vs null mean 0.0285, z=-2.94, p=0.00999, q=0.0719)
- **exact_example_target_yes** — Yes-rate on trials whose target is one of the shown examples, minus yes-rate on targets within 1-3 of the nearest example; observed > null means people treat a repeated example as near-certainly produced far more than nearby numbers, and the model under-produces this boost. (observed 0.371 vs null mean 0.304, z=2.44, p=0.018, q=0.0719)

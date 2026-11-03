# Critique of `repdigit_pattern_chart_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **singleton_near_far_gradient** — For single-example sets, yes rate to targets within 10 of the example minus yes rate to targets more than 10 away; observed above null means people generalise more locally (steeper similarity gradient) than the model predicts, below means flatter. (observed 0.0596 vs null mean 0.235, z=-2.08, p=0.036, q=0.288)

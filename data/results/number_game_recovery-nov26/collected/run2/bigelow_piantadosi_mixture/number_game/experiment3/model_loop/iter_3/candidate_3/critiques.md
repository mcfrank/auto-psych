# Critique of `heading_weber_upward_with_replacement`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prefix_extension_yes_change** — For 4-example sets whose first three examples are also shown as a 3-example set, yes rate on the 4-example set minus the 3-example set over the same non-example targets; observed above null means adding a fourth example narrows generalization less (or broadens it more) than the model predicts. (observed 0.287 vs null mean 0.106, z=2.51, p=0.014, q=0.112)

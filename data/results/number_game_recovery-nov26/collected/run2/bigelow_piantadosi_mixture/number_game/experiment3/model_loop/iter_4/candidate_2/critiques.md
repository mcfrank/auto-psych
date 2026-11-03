# Critique of `position_decaying_example_evidence`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **fourth_example_prefix_broadening** — For 4-example sets whose first three examples are also shown as a 3-example set (31,3,1 -> 31,3,1,15), yes rate on the 4-example set minus the 3-example set over shared non-example targets; observed above null means the fourth example broadens generalization more (narrows less) than the model's position-decaying evidence predicts. (observed 0.287 vs null mean 0.106, z=2.43, p=0.016, q=0.128)

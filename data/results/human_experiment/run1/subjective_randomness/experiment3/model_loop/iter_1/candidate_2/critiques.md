# Critique of `linear_accumulated_typicality`

2 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **length_penalty_scaling** — Mean choice rate of the shorter sequence when both sequences are highly imbalanced (>0.4), testing if the penalty accumulates linearly. (observed 0.302 vs null mean 0.378, z=-3.09, p=0.0199, q=0.159)
- **extreme_proportion_penalty** — Mean choice rate of sequence A when it has extreme proportion imbalance (>0.6) and sequence B is balanced (<0.2). (observed 0.13 vs null mean 0.205, z=-2.22, p=0.0398, q=0.159)

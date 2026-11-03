# Critique of `example_anchored_exception_neighbourhood`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **outlier_vs_majority_neighbour** — In 3-4 example sets where all but one example share a final digit, yes-rate for non-example targets within 3 of the outlier example minus yes-rate for off-digit non-example targets within 3 of a majority example (and not near the outlier); observed above the null means people favour the outlier's neighbourhood even more than the anchored-exception model, below means the model over-produces it. (observed 0.119 vs null mean -0.00588, z=2.95, p=0.004, q=0.032) [survives FDR]

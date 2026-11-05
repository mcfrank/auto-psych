# Critique of `range_bounded_parity_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_generalization_rate** — Yes rate for non-example targets on sets with exactly one example; observed above null means people generalize more broadly from a single example than the model's size principle allows (below = more conservative). (observed 0.58 vs null mean 0.384, z=7.06, p=0.002, q=0.016) [survives FDR]

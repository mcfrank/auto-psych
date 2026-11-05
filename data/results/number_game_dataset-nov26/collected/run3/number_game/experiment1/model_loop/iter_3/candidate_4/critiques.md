# Critique of `parity_restricted_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **monotone_order_shared_digit_contrast** — Among 3-example sets whose examples all share a last digit, yes rate of non-example targets lacking that digit for sets shown in monotone (sorted ascending/descending) order minus sets shown in non-monotone order; observed above null means presentation order (an orderly sequence) loosens the "ends in d" rule, which the order-blind model cannot produce (below null means the reverse). (observed 0.198 vs null mean -0.00688, z=5.14, p=0.002, q=0.016) [survives FDR]

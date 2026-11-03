# Critique of `capacity_limited_evidence_integration`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **mixed_parity_range_gap** — In sets of >=2 examples mixing odd and even numbers, yes rate for targets inside [min,max] of the examples minus targets outside it; observed above null means people use the example range more sharply than the model predicts. (observed 0.207 vs null mean 0.132, z=3.00, p=0.002, q=0.016) [survives FDR]

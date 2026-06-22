# Critique of `prototype_similarity`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **extreme_imbalance_aversion** — The probability of choosing the perfectly imbalanced sequence (all H or all T) when paired against a non-extreme sequence. (observed 0.45 vs null mean 0.247, z=2.74, p=0.00995, q=0.0597)

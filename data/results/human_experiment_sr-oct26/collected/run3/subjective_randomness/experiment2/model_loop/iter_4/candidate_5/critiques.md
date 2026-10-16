# Critique of `second_order_motif_side_habit`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **heads_count_preference** — Among trials whose sequences have different numbers of H, the proportion choosing the sequence with more heads; the model is H/T symmetric, so observed > null means people favour head-heavy sequences as random, observed < null that they favour tail-heavy ones (a label asymmetry the model cannot produce). (observed 0.473 vs null mean 0.488, z=-2.93, p=0.004, q=0.032) [survives FDR]

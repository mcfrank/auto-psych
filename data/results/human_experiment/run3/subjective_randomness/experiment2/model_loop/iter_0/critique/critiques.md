# Critique of `smoothed_prototype_distance`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **alt_motifs_preference** — The response rate for sequence A when it has more alternating motifs than sequence B, conditional on similar overall bigram alternation rates. (observed 0.595 vs null mean 0.5, z=2.35, p=0.0398, q=0.279)

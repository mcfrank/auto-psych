# Critique of `prototype_similarity`

3 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **rep_motif_aversion** — Rate of choosing the sequence with fewer repetition motifs per item, when alternation rates are similar. (observed 0.375 vs null mean 0.448, z=-3.10, p=0.00995, q=0.0796)
- **heads_tails_asymmetry** — Rate of choosing the sequence with more heads, when imbalance is identical but they skew in opposite directions. (observed 0.275 vs null mean 0.41, z=-2.48, p=0.0398, q=0.133)
- **length_preference** — The overall rate of choosing the longer sequence when sequence lengths differ. (observed 0.619 vs null mean 0.594, z=1.83, p=0.0498, q=0.133)

# Critique of `heavy_tailed_signed_sensitivity_motif`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **more_heads_choice_rate** — Among pairs whose sequences differ in heads count, the rate of choosing the sequence with more heads; observed below null means people avoid head-heavy sequences more than the rigged-trick-coin prior predicts, above means the model over-penalises head-heavy sequences. (observed 0.485 vs null mean 0.498, z=-2.42, p=0.012, q=0.0959)
- **short_pairs_more_switches** — Among pairs of length 2-3 with differing switch counts, the rate of choosing the sequence with more switches; observed below null means people prefer alternation in very short sequences less than the model's length-scaled evidence predicts, above means more. (observed 0.744 vs null mean 0.776, z=-1.98, p=0.05, q=0.2)

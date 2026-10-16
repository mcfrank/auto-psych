# Critique of `second_order_chance_motif_suspicion`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **heads_count_preference** — Among trials whose sequences have different numbers of H, the proportion choosing the sequence with more heads; the model is H/T symmetric, so observed > null means people favour head-heavy sequences as random (a label asymmetry the model cannot produce), < null tail-heavy. (observed 0.473 vs null mean 0.488, z=-2.96, p=0.00799, q=0.04) [survives FDR]
- **participant_side_bias_spread** — SD across participants of each person's proportion of Left choices; observed > null means the model (no side/response bias) under-produces person-level left/right response biases. (observed 0.0756 vs null mean 0.0643, z=2.60, p=0.00999, q=0.04) [survives FDR]
- **response_side_perseveration** — Proportion of consecutive trials (within participant, ordered by trial_index) on which the same side (Left/Right) is chosen as on the previous trial; observed > null means the model under-produces motor/side perseveration, < null that people alternate response sides more than the stimulus-driven model predicts. (observed 0.513 vs null mean 0.499, z=2.26, p=0.024, q=0.0639)

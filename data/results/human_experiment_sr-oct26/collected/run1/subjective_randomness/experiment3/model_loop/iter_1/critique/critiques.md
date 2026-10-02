# Critique of `asymmetric_tolerance_lopsided_stretch`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_side_bias_variance** — Variance across participants of each participant's proportion of LEFT choices; observed above null_mean means individuals carry personal side (button) biases the model, which has no per-person side term, under-produces. (observed 0.00556 vs null mean 0.00432, z=2.56, p=0.016, q=0.128)

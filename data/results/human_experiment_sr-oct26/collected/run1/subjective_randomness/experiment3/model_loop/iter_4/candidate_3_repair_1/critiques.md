# Critique of `personal_pattern_detection_gain`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_majority_agreement_variance** — Variance across participants of each participant's rate of agreeing with the pair's overall majority choice (pair identified regardless of side); observed above null_mean means people differ in how consistently they follow the consensus more than the model's personal gain, sensitivity and lapse terms produce, below means the model now over-disperses individuals. (observed 0.026 vs null mean 0.0213, z=2.84, p=0.00599, q=0.048) [survives FDR]
- **participant_side_bias_variance** — Variance across participants of each participant's proportion of Left choices; observed above null_mean means people carry personal left/right button biases that the model (no side term) under-produces. (observed 0.00551 vs null mean 0.00428, z=2.58, p=0.016, q=0.0639)

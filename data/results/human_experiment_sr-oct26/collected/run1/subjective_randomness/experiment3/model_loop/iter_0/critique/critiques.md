# Critique of `most_lopsided_stretch_aversion`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_side_bias_sd** — Between-participant standard deviation of each person's left-choice rate; observed above null_mean means individuals have stable personal left/right response biases that the model (no side term) under-produces. (observed 0.0746 vs null mean 0.0654, z=2.45, p=0.018, q=0.144)
- **left_choice_rate** — Overall proportion of trials on which the LEFT sequence was chosen; observed above the model's null_mean means people have a left-side response bias the model (which has no side term) under-produces, below means a right bias. (observed 0.509 vs null mean 0.499, z=2.13, p=0.046, q=0.184)

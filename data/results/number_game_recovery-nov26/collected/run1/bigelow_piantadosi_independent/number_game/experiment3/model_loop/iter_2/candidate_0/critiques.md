# Critique of `anchored_neighbourhood_side_channel`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **mixed_parity_even_target_bias** — In sets whose examples include both odd and even numbers, yes-rate for even non-example targets minus odd non-example targets; observed above the null means people favour even targets beyond what the model's (parity-symmetric given mixed examples) rules and intervals predict, below means the model over-produces an even bias. (observed 0.0896 vs null mean 0.0546, z=2.23, p=0.034, q=0.256)

# Critique of `digit_coincidence_noticing_guess`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_same_last_digit_far** — Yes-rate for single-example sets on targets >=20 away sharing the example's last digit; observed above null means the model under-produces digit generalisation from one example. (observed 0.333 vs null mean 0.505, z=-2.27, p=0.036, q=0.288)

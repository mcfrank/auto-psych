# Critique of `erlang_interval_prior_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_digit_far_targets** — For targets more than 15 beyond the examples' range, yes-rate when the target shares a last digit with some example minus when it shares none; observed > null means people use surface digit similarity more than the model's rules capture. (observed 0.108 vs null mean 0.173, z=-2.02, p=0.04, q=0.232)

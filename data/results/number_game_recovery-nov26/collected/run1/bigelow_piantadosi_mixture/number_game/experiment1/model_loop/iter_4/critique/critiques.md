# Critique of `sticky_guess_power_law_ranges`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_last_digit_far** — For targets more than 15 outside the examples' range, yes-rate when the target shares its last digit with some example minus when it shares none; observed < null means the model over-generalises by "ends in d" rules relative to people, observed > null means people use last-digit similarity more. (observed 0.108 vs null mean 0.186, z=-2.43, p=0.022, q=0.176)

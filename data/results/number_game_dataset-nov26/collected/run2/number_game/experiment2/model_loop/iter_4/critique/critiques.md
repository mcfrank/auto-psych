# Critique of `progression_band_first_digit_primacy`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **inrange_nonshared_rate** — In multi-example sets, yes-rate for targets strictly inside the examples' [min,max] range that share neither final digit, parity nor seed-multiple with the examples, minus the same for targets outside the range; observed > null means the model under-produces pure magnitude ("in the range") generalisation. (observed 0.0662 vs null mean -0.0144, z=2.66, p=0.014, q=0.112)

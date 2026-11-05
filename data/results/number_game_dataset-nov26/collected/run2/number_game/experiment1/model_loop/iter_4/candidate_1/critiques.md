# Critique of `one_sided_magnitude_bound_overlap`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_final_digit_match_rate** — Yes-rate for targets that end in the final digit shared by all (>=2) examples (e.g. 28,98 -> 38, 58); observed > null means the model under-endorses targets fitting a shared last-digit rule (treats it as just one diluted property), observed < null means it over-endorses them. (observed 0.819 vs null mean 0.677, z=2.57, p=0.00799, q=0.0639)

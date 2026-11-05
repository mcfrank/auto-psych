# Critique of `round_number_kinship_exemplar`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **two_digit_example_multiple_contrast** — Among targets >=10 from every example and sharing no example's final digit, yes-rate for targets that are multiples of some example >=13 but of no example 2-12, minus yes-rate for targets that are not multiples, divisors or gcd>=3 kin of any example; observed < null means the model's size-independent multiples route still over-credits multiples of large two-digit examples (a times-table effect), observed > null means it under-credits them. (observed 0.0608 vs null mean 0.219, z=-2.50, p=0.00999, q=0.0799)

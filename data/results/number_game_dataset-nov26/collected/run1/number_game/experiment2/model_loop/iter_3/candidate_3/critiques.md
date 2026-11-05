# Critique of `factor_kin_multiple_route`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **large_example_multiple_contrast** — Among targets >=10 from every example and sharing no example's final digit, yes-rate for targets that are multiples only of an example >=13 (no example 2-12 divides them; e.g. 94 after 47) minus yes-rate for targets that are not multiples, divisors or factor-kin of any example; observed < null means the model's size-independent multiples route over-credits multiples of large examples (people's multiple route is a times-table effect for small numbers), observed > null means it under-credits them. (observed 0.0608 vs null mean 0.216, z=-2.38, p=0.018, q=0.144)

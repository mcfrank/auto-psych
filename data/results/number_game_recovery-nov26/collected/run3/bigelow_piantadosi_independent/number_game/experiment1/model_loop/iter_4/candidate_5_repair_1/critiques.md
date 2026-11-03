# Critique of `multiplicative_not_exponential_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_contained_digit_other_ending_yes** — For 2+ example sets in which every example contains some digit d, yes rate on non-example targets that contain such a d but whose last digit matches no example's last digit; observed > null means people use "contains digit d" more than the model predicts, observed < null means less. (observed 0.433 vs null mean 0.61, z=-5.16, p=0.002, q=0.016) [survives FDR]
- **upward_minus_downward_near** — For non-example targets within 15 of their nearest example, yes rate when the target is above that example minus when it is below; observed > null means people generalise more upward than the model's roughly symmetric intervals predict, observed < null means more downward. (observed 0.0329 vs null mean -0.00841, z=2.12, p=0.038, q=0.152)

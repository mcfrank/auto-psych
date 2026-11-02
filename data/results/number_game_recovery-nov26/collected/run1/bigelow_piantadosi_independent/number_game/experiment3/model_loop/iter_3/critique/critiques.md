# Critique of `gamma_shaped_range_width_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_contained_digit_tens** — In 2-4 example sets where every example contains a common digit d, among non-example targets more than 5 from every example and sharing no final digit with any example, yes-rate for targets whose tens digit is such a d minus other such targets; observed above the null means people use 'contains digit d' rules more than the model, below means the model over-weights those rules. (observed 0.565 vs null mean 0.294, z=2.35, p=0.026, q=0.208)

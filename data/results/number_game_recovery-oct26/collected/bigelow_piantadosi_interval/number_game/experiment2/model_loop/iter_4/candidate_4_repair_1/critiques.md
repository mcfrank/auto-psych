# Critique of `volatile_concept_changepoint`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **power_of_two_specificity** — Among sets whose examples are all powers of 2 (2..64), yes rate to non-example powers of 2 minus yes rate to other even non-example targets; observed below the null means people generalise to the remaining powers of 2 less sharply (relative to evens) than the model, above means the model under-commits to the specific rule. (observed -0.34 vs null mean -0.149, z=-2.40, p=0.02, q=0.16)

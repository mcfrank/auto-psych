# Critique of `family_weighted_concept_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **digit_set_inrange_nonmatching_yes_rate** — Yes-rate for targets inside [min-5,max+5] of sets whose >=2 examples all share a final digit, when the target does NOT end in that digit; observed < null means the model over-produces interval/parity generalization and under-weights the "ends in d" rule people adopt for such sets. (observed 0.0177 vs null mean 0.0383, z=-2.32, p=0.02, q=0.16)

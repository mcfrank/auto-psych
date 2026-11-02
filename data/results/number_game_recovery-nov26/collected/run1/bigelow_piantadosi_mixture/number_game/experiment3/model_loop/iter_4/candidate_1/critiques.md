# Critique of `span_extrapolation_hesitant_guess_2`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **first_example_side_extension** — For sets of >=2 examples whose first example is the min or the max, yes rate for targets 1-20 beyond the span on the first example's side minus on the opposite side; observed > null means people anchor extrapolation on the first example more than the model's sticky first guess predicts, observed < null they favour the later examples' side. (observed -0.00875 vs null mean 0.0566, z=-2.27, p=0.022, q=0.176)

# Critique of `run_length_variety_preference`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **high_alternation_pair_choice_rate** — Among trials where both sequences have alternation rate >= 0.6 and the rates differ, the proportion choosing the MORE alternating sequence; observed above the null means people penalise over-alternation less than the model's quadratic ideal-rate term implies, below means they penalise it more. (observed 0.594 vs null mean 0.51, z=2.69, p=0.00999, q=0.0799)

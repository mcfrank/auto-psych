# Critique of `individual_alternation_sensitivity_mirror`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **high_alternation_pair_choice_rate** — Among trials where both sequences have alternation rate >= 0.6 and the rates differ, the proportion choosing the MORE alternating sequence; observed above the null means people penalise over-alternation less than the model's quadratic ideal-rate term implies, below means they penalise it more. (observed 0.594 vs null mean 0.477, z=3.89, p=0.002, q=0.016) [survives FDR]
- **run_length_variety_choice_rate** — Among trials where the two sequences have the same alternation rate but a different number of distinct run lengths, the proportion choosing the sequence with MORE distinct run lengths; observed above the null means people prize irregular run structure beyond what the model's alternation/imbalance/streak/symmetry terms capture, below means they prefer regular run structure more than the model predicts. (observed 0.706 vs null mean 0.624, z=3.38, p=0.004, q=0.016) [survives FDR]

# Critique of `person_specific_chance_switch_belief`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_side_bias_spread** — SD across participants of each person's proportion of Left choices; observed > null means the model (which has no side/response bias) under-produces person-level left/right response biases. (observed 0.0756 vs null mean 0.0642, z=2.62, p=0.014, q=0.112)

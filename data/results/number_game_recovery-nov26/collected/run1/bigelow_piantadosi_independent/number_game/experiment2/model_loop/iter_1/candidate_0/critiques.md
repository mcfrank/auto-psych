# Critique of `final_digit_salient_rules`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **example_target_yes_rate** — Yes rate when the target is itself one of the shown examples; observed < model means people reject examples more often than the model's near-certain membership implies (a response-level lapse). (observed 0.936 vs null mean 0.874, z=2.22, p=0.034, q=0.272)

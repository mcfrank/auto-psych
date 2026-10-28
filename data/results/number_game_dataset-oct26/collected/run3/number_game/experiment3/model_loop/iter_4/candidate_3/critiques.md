# Critique of `outlier_tolerant_rule_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **tens_sets_nonround_targets** — For sets of >= 2 examples that are all multiples of 10, yes-rate for targets that are not multiples of 10; observed > null means people accept non-round numbers for all-round sets more than the model's strong multiples-of-10 rule predicts (observed < null: they reject them more). (observed 0.0175 vs null mean 0.0815, z=-2.65, p=0.00599, q=0.048) [survives FDR]
- **single_digit_targets_for_high_sets** — For sets whose smallest example is >= 20, yes-rate for single-digit targets (1-9); observed > null means people accept small one-digit numbers for high-valued sets more than the model's rules/intervals predict (observed < null: they treat one-digit numbers as a separate category and reject them more than the model). (observed 0.147 vs null mean 0.174, z=-2.21, p=0.026, q=0.104)

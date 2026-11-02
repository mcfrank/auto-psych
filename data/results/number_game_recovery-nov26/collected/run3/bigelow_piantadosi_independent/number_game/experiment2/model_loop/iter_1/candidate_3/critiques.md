# Critique of `factor_graded_multiples_concepts`

1 of 7 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **yes_trial_index_slope** — OLS slope of yes (0/1) on trial_index (per 100 trials); observed more negative than null means people drift toward saying no over the session, which the stimulus-only model does not produce. (observed -0.053 vs null mean -0.0418, z=-2.16, p=0.03, q=0.21)

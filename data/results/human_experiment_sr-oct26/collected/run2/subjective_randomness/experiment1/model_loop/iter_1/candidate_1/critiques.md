# Critique of `personal_alternation_ideal`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **periodic_penalty_given_alt** — OLS coefficient of chose_left on the periodicity-indicator difference (right periodic minus left periodic, e.g. HTHTHTHT, HHTTHHTT), controlling for alternation and squared-distance differences; t_observed > null_mean means people reject regular repeating patterns more than the model predicts. (observed 0.0824 vs null mean 0.0129, z=6.08, p=0.002, q=0.00799) [survives FDR]
- **longest_run_coef_given_alt** — OLS coefficient of chose_left on the longest-run-length difference (normalised by length, right minus left), controlling for alternation difference and squared-distance difference; t_observed > null_mean means long streaks are penalised beyond what alternation rate explains. (observed -0.172 vs null mean 0.0302, z=-3.52, p=0.002, q=0.00799) [survives FDR]
- **perfect_alternation_choice** — Among trials where exactly one sequence is perfectly alternating (HTHT.../THTH...), the proportion choosing that sequence; t_observed < null_mean means the model over-predicts preference for perfect alternation (people see it as too regular). (observed 0.634 vs null mean 0.683, z=-2.50, p=0.014, q=0.0373) [survives FDR]
- **left_choice_rate** — Overall proportion of Left choices; t_observed > null_mean means a left-side response bias the side-symmetric model does not produce. (observed 0.518 vs null mean 0.502, z=2.19, p=0.034, q=0.0679)
- **equal_alt_balance_pref** — Among trials whose two sequences have identical alternation rates, the proportion choosing the sequence with the more balanced H/T count (ties in balance excluded); the model predicts 0.5, so t_observed > null_mean means people use H/T balance, which the model ignores. (observed 0.613 vs null mean 0.501, z=1.96, p=0.05, q=0.0799)

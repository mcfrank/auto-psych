# Critique of `dual_route_factor_and_digit_rule_exemplar`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **session_drift_late_minus_early** — Yes-rate on trials in the last third of the session (trial_index >= 300) minus the first third (trial_index < 150); observed above the null means people drift toward "yes" over the session, a nonstationarity the stationary model cannot produce (below means drift toward "no"). (observed 0.0689 vs null mean 0.0154, z=3.39, p=0.002, q=0.016) [survives FDR]

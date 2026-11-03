# Critique of `effortful_computed_rule_search`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **person_session_drift_sd** — SD across participants of (yes rate in second half of session minus first half, by trial_index); observed above null means individuals drift in yes-propensity over the session in different directions, which the model's constant per-person parameters cannot produce. (observed 0.0253 vs null mean 0.0407, z=-1.95, p=0.034, q=0.272)

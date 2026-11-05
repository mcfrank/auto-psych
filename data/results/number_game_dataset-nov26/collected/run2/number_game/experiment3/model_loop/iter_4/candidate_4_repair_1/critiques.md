# Critique of `parity_doubt_three_example_memory`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **early_session_yes_gap** — Yes rate on trials with trial_index < 118 (first quarter of the session) minus yes rate on later trials; observed < null means people start out more conservative than later in the session, a warm-up drift the stationary model cannot produce (observed > null: they start more liberal). (observed -0.0351 vs null mean 0.00387, z=-3.16, p=0.004, q=0.032) [survives FDR]

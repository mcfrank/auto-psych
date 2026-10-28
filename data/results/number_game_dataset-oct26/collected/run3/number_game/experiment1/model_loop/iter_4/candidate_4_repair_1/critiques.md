# Critique of `decade_aligned_range_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **last_minus_first_quartile_yes** — Yes-rate on trials in the top quartile of trial_index minus yes-rate in the bottom quartile; the model has no session dynamics (null ~0), so observed above null means people grow more liberal over the session, below means more conservative. (observed 0.067 vs null mean 0.00444, z=2.38, p=0.022, q=0.176)

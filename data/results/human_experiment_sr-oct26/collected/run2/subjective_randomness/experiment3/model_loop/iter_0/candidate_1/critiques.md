# Critique of `tally_span_switch_ideal_periodic_unit`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **item_rate_overdispersion** — Across distinct unordered pairs, the variance of the rate at which the alphabetically-first sequence is chosen (weighted by n); observed above null means items are more extreme/polarised than the model predicts (it misses stimulus features driving strong consensus), below null means it is too confident. (observed 0.0707 vs null mean 0.0662, z=1.92, p=0.05, q=0.288)

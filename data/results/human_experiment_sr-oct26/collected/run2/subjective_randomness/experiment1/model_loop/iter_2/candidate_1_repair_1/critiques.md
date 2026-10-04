# Critique of `periodic_penalized_alternation_ideal`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **alternation_pref_by_length_slope** — Slope over sequence length of the rate of choosing the sequence with more switches (pairs with unequal switch counts); observed above null_mean means the model under-predicts how the higher-switch preference grows with length. (observed -0.0435 vs null mean -0.0585, z=2.92, p=0.00599, q=0.048) [survives FDR]
- **shorter_longest_run_pref** — Among pairs with equal switch counts but different longest-run lengths, rate of choosing the sequence with the shorter longest run; observed above null_mean means the model under-penalises long streaks beyond switch rate. (observed 0.588 vs null mean 0.451, z=2.45, p=0.026, q=0.104)

# Critique of `lapsing_growing_evidence_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **late_minus_early_far_target_yes** — Yes-rate on far targets (>20 from every example) with trial_index above the median minus at/below it; observed above null means blind/loose "yes" to concept-excluded targets grows over the session, which a constant lapse rate cannot produce. (observed 0.0897 vs null mean 0.0077, z=2.36, p=0.022, q=0.168)
- **same_decade_outside_bonus** — Among outside-range targets at distance 3-15 from the nearest example, yes-rate when the target shares its tens decade with some example minus when it does not; observed above null means people group numbers by decade more than the model predicts. (observed 0.149 vs null mean 0.0165, z=2.03, p=0.042, q=0.168)

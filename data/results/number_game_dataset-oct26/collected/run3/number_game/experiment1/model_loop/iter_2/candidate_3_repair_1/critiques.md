# Critique of `sample_size_growing_evidence_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **late_minus_early_yes_rate** — Yes-rate on trials with trial_index above the median minus yes-rate at or below it; the model has no session dynamics, so observed above null means people become more liberal (more yes) later in the session, below means more conservative. (observed 0.0429 vs null mean 0.00153, z=2.25, p=0.034, q=0.272)

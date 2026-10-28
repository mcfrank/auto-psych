# Critique of `outlier_tolerant_rule_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **within_cell_trial_drift** — Correlation between rating and trial_index after removing each (set, target) cell's mean from both; observed > null (positive) means people say "yes" more often later in the session for the same stimulus, a drift toward liberal responding the stationary model cannot produce (observed < null: more conservative over time). (observed 0.0272 vs null mean -0.000137, z=2.24, p=0.022, q=0.176)
- **prime_sets_far_prime_targets** — For 2+ example sets whose examples are all primes, yes-rate for prime non-example targets lying more than 10 outside the set's [min,max] range; observed > null means people extend a "primes"/"odd" concept beyond the examples' range more than the model (whose tight intervals out-compete the 26-member prime rule) predicts (observed < null: less). (observed 0.348 vs null mean 0.459, z=-2.13, p=0.044, q=0.176)

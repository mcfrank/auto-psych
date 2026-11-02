# Critique of `graded_ratio_ceiling_criterion`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **new_set_first_trial_yes_gap** — Yes rate on non-example targets on the first trial after the set changes (or session start, trial_index 0) minus yes rate on subsequent trials with the same set as the previous trial; nonzero observed vs a null near the stimulus-driven value means a set-novelty effect the stationary model lacks (positive = more liberal on the first query about a new set). (observed -0.24 vs null mean -0.209, z=-2.03, p=0.038, q=0.304)

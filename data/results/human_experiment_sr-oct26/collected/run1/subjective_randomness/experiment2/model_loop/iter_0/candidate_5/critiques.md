# Critique of `individual_streak_aversion_lapse`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **terminal_run_choice_rate** — Among pairs whose sequences differ in the length of the run at their end (final streak), the proportion of choices of the sequence with the longer final run; observed < null means people penalise streaks at the end of a sequence (recency) more than the model's position-blind longest-run term predicts. (observed 0.361 vs null mean 0.345, z=2.07, p=0.038, q=0.221)

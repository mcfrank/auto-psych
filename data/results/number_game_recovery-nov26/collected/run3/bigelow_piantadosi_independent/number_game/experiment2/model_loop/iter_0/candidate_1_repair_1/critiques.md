# Critique of `costly_squares_final_digit_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_rating_correlation** — Pearson correlation between trial_index and rating across all trials; observed more negative than null means people say yes less often as the session goes on, beyond what the stimulus order implies, and the model has no such drift. (observed -0.143 vs null mean -0.107, z=-2.54, p=0.00599, q=0.048) [survives FDR]

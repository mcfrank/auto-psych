# Critique of `bigelow_piantadosi_interval`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_drift** — Pearson correlation between trial_index and the yes response; the model has no time-on-task term so predicts ~0, and an observed value away from null_mean means people drift toward more (positive) or fewer (negative) yeses over the session. (observed 0.0495 vs null mean 0.00734, z=2.30, p=0.038, q=0.304)

# Critique of `bigelow_piantadosi_independent`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **recency_last_vs_first_example** — Among multi-example trials, yes-rate when the target is numerically closer to the last-shown example minus when closer to the first-shown; the model is order-invariant, so observed above null means a recency effect (below null a primacy effect). (observed -0.0459 vs null mean -0.121, z=3.52, p=0.002, q=0.016) [survives FDR]
- **single_example_yes_rate** — Yes-rate on trials with a single example (excluding target == example); observed above null means humans generalise more broadly from one example than the model's size-principle posterior does. (observed 0.525 vs null mean 0.46, z=2.44, p=0.018, q=0.0719)
- **trial_index_yes_correlation** — Pearson correlation between trial_index and the yes response; the model has no time dependence, so observed away from null means yes-rate drifts across the session (positive = more yes later, negative = fewer). (observed 0.0417 vs null mean 0.00422, z=2.03, p=0.046, q=0.123)

# Critique of `evidence_scarce_guessing_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_yes_slope** — OLS slope of yes (0/1) on trial_index (per 100 trials); observed > model means people drift toward saying "yes" more as the session goes on, which the trial-invariant model cannot produce, observed < model means a drift toward "no". (observed 0.0193 vs null mean 0.00437, z=3.25, p=0.004, q=0.024) [survives FDR]
- **parity_mismatch_yes_rate** — Among sets of 2+ examples that all share one parity, the yes-rate for targets of the opposite parity; observed > model means people use parity less strictly (generalize across parity more) than the model predicts, observed < model means they reject parity violators more sharply. (observed 0.0979 vs null mean 0.122, z=-2.39, p=0.00599, q=0.024) [survives FDR]

# Critique of `outlier_tolerant_rule_concepts`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **late_minus_early_yes_rate** — Yes-rate on trials with trial_index >= 300 minus yes-rate on trials with trial_index < 150; observed > model means people drift toward "yes" over the session more than the trial-invariant model predicts (observed < model means drift toward "no"). (observed 0.0558 vs null mean 0.0112, z=3.02, p=0.004, q=0.032) [survives FDR]
- **cell_consensus_all** — Over all (set, target) cells with at least 5 responses, the mean of p*(1-p) where p is the cell's yes-rate; observed < model means people's answers to the same item are more consensual / all-or-none than the model's graded probabilities produce, observed > model means more within-item disagreement than predicted. (observed 0.148 vs null mean 0.157, z=-2.87, p=0.00999, q=0.04) [survives FDR]
- **parity_mismatch_yes_rate** — Among sets of 2+ examples that all share one parity, the yes-rate for targets of the opposite parity; observed > model means people use parity less strictly than the model predicts, observed < model means they reject parity violators more sharply than the model's rule/interval mixture with guessing allows. (observed 0.0979 vs null mean 0.122, z=-2.43, p=0.024, q=0.0639)

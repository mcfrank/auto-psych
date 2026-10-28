# Critique of `evidence_scarce_guessing_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **cell_response_graded_ness_multi** — Over (set, target) cells from sets of 2+ examples, the mean of p*(1-p) where p is the cell's yes-rate across its repeated responses; observed < model means people's answers per item are more consensual/all-or-none than the model's graded posterior-mixture probabilities produce, observed > model means more within-item disagreement than predicted. (observed 0.142 vs null mean 0.152, z=-2.92, p=0.004, q=0.032) [survives FDR]
- **near_rule_violator_yes** — For sets of 2+ examples all divisible by 3, 4, 5 or 10 (or all sharing a last digit), yes-rate on non-example targets within 5 of the nearest example that violate every such shared rule; observed > model means numeric proximity pulls people to "yes" despite a rule violation more than the model predicts, observed < model means people reject nearby rule violators more sharply. (observed 0.201 vs null mean 0.252, z=-2.19, p=0.024, q=0.0959)

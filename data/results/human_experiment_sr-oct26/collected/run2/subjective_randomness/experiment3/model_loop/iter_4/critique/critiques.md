# Critique of `terminal_streak_recency_pattern_rule`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **lag2_repeat_equal_switch** — Among pairs with equal switch counts that differ in the number of 4-flip strictly alternating windows (HTHT/THTH), the rate of choosing the sequence with fewer such windows; observed above null means people see extended alternation as patterned more than the model predicts (model under-penalises local alternation), below means less. (observed 0.5 vs null mean 0.449, z=2.13, p=0.038, q=0.304)

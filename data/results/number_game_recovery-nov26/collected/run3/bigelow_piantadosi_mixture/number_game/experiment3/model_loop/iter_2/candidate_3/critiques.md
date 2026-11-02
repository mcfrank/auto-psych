# Critique of `variable_rule_range_expectation`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **tight_rule_inhull_nonmember_yes** — Yes-rate for rule NON-members inside the examples' span in multi-example sets that share a multiple-of-k (k>=2) or odd rule and span <=40 (e.g. 37,57 -> 44); observed > null means people fill in narrow spans with off-rule numbers more than the model (the open span-contrast critique, re-tested as a level). (observed 0.523 vs null mean 0.41, z=2.70, p=0.004, q=0.032) [survives FDR]

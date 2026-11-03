# Critique of `rule_count_familiar_strays`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **clean_rule_member_inside_minus_outside_hull** — In >=2-example sets whose examples all belong to some narrow rule (<=25 members), yes rate for non-example members of the smallest such rule lying inside the examples' range minus those outside it; observed above null means people restrict a rule to the examples' magnitude range more than the model (range only via separate interval hypotheses), below means less. (observed 0.179 vs null mean 0.1, z=2.17, p=0.05, q=0.288)

# Critique of `outlier_tolerant_rule_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ruleless_sets_in_range_yes** — For 3-4 example sets with no common factor, no shared last digit and mixed parity (sets the model can only explain with an interval), yes-rate for non-example targets strictly inside the [min,max] range; observed < null means people do not fill the gaps of patternless sets as an interval learner would, treating them as random (observed > null: they fill the range more than the model). (observed 0.423 vs null mean 0.368, z=2.36, p=0.032, q=0.256)

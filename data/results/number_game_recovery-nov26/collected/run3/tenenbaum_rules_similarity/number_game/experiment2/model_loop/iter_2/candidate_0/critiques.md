# Critique of `beta_share_width_exception_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **nested_multiple_specificity** — For sets of >=2 examples whose largest common divisor m in 3..12 has a proper divisor d>1 (e.g. all multiples of 9 -> d=3), yes rate on non-example multiples of m minus non-example multiples of d that are not multiples of m; observed > null means people prefer the most specific multiplicative rule more strongly than the model's size principle predicts, observed < null means they generalise to the broader rule more. (observed 0.495 vs null mean 0.388, z=2.04, p=0.042, q=0.282)

# Critique of `size_principle_strength_family_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **lure37_yes_rate_power2_sets** — Yes-rate for target 37 on sets whose examples are all powers of 2 (37 is only in the 'powers of 2, or 37' rule); observed < null means the model over-produces yes to the 37 lure by giving that ad-hoc rule as much prior mass as plain powers of 2. (observed 0.0333 vs null mean 0.283, z=-3.10, p=0.00599, q=0.048) [survives FDR]

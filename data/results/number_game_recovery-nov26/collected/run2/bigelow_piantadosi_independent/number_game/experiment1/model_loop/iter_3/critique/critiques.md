# Critique of `even_odd_split_family_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **last_vs_first_example_proximity** — Yes rate for non-example targets within 5 of the LAST-shown example but >5 from the first-shown, minus the rate for targets within 5 of the FIRST-shown but >5 from the last (multi-example sets); the model is order-invariant, so t_observed above null_mean means recency (last example weighted more), below means primacy. (observed 0.068 vs null mean -0.0325, z=2.64, p=0.018, q=0.144)

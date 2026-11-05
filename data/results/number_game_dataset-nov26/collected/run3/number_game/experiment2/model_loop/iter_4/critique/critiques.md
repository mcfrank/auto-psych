# Critique of `comparison_dependent_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ruleless_set_in_range_rate** — For 2+ example sets that no simple rule covers (no shared parity, last digit, contained digit, multiple of 3-12, square, cube or power of 2), yes rate on non-example targets inside the examples' range; observed above null means the model under-predicts interval-like generalization when no rule fits (below: over-predicts). (observed 0.491 vs null mean 0.426, z=2.88, p=0.00599, q=0.048) [survives FDR]
- **single_example_multiples_rate** — Yes rate on one-example sets (example x>=2) for non-example targets that are multiples of x; observed above null means the model still under-generalizes a lone example to its arithmetic rule (6->18, 25->100). (observed 0.725 vs null mean 0.629, z=2.21, p=0.038, q=0.152)

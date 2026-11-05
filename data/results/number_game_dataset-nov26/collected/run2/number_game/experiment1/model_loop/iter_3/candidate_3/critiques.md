# Critique of `decimal_multiple_salience_overlap`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **above_vs_below_span_asymmetry** — Yes-rate for targets larger than every example minus yes-rate for targets smaller than every example (multi-example sets); the model's span term is symmetric, so observed > null means people extend upward more readily than downward (observed < null: downward more readily). (observed 0.0968 vs null mean -0.025, z=3.77, p=0.002, q=0.016) [survives FDR]
- **set_size_parity_consistent_gap** — Among parity-homogeneous sets, yes-rate for parity-consistent targets in 3-4-example sets minus in 1-2-example sets; observed < null means people narrow generalization with more examples (a size-principle effect the model lacks), observed > null means more examples make people more permissive than the model predicts. (observed -0.0852 vs null mean 0.0157, z=-3.00, p=0.004, q=0.016) [survives FDR]

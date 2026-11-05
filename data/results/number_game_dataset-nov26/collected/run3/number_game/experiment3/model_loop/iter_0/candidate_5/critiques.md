# Critique of `consistent_examples_bounded_rules`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ruleless_set_in_range_rate** — On 2+ example sets that no rule (odd/even, multiples 3-12, powers, squares, cubes, primes, ends-in/contains digit) fits fully, yes rate for in-range non-example targets; observed above null means the model under-endorses the range when no rule explains every example (too much belief kept on near-fitting rules). (observed 0.489 vs null mean 0.403, z=4.07, p=0.002, q=0.016) [survives FDR]
- **in_minus_out_range_two_examples** — On 2-example sets, yes rate for non-example targets strictly inside the examples' range minus yes rate for targets outside it; observed above null means the model under-weights range/interval generalization relative to rules for pairs. (observed 0.204 vs null mean 0.14, z=2.88, p=0.00599, q=0.024) [survives FDR]

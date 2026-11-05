# Critique of `range_specific_evidence_strength`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ruleless_set_out_of_range_rate** — On 2+ example sets that no rule (odd/even, multiples 3-12, powers 2-10, squares, cubes, primes, ends-in/contains digit) fits fully, yes rate for non-example targets outside the examples' range; observed below null means the model leaks too much generalization beyond the range when no rule explains the examples. (observed 0.265 vs null mean 0.345, z=-2.18, p=0.036, q=0.186)

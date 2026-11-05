# Critique of `range_specific_evidence_strength`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_digit_set_nonmatching_same_parity** — On 3-4 example sets whose examples all share a last digit, yes rate for non-example targets with a different last digit but the same parity as the examples; observed below null means the model leaks too much generalization to same-parity numbers (via odd/even, contains-digit, prime rules) when an ends-in-digit rule fits. (observed 0.0862 vs null mean 0.187, z=-4.40, p=0.002, q=0.016) [survives FDR]
- **ruleless_three_plus_in_range_rate** — On 3-4 example sets that no rule (odd/even, multiples 3-12, powers 2-10, squares, cubes, primes, ends-in/contains digit) fits fully, yes rate for in-range non-example targets; observed above null means the model under-generalizes within the examples' range when no rule explains the examples. (observed 0.484 vs null mean 0.423, z=2.66, p=0.018, q=0.0719)

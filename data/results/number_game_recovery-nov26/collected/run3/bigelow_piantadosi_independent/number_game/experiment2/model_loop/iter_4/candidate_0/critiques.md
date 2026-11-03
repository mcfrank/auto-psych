# Critique of `prime_over_odd_reading`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **fives_set_mult5_target_lift** — In sets of >=2 examples that are all multiples of 5 (not all multiples of 10), yes rate on non-example multiples-of-5 targets minus yes rate on non-multiple-of-5 targets; observed > null means the model under-generalises "multiples of 5". (observed 0.775 vs null mean 0.541, z=4.86, p=0.002, q=0.016) [survives FDR]
- **mult3_set_mult3_target_lift** — In sets of >=2 examples all divisible by 3 (and not all divisible by 5), yes rate on non-example multiple-of-3 targets minus yes rate on non-multiple-of-3 targets; observed < null means the model over-generalises "multiples of 3". (observed -0.0392 vs null mean 0.0448, z=-2.76, p=0.004, q=0.016) [survives FDR]
- **repdigit_set_repdigit_target_lift** — In sets containing >=2 repdigit examples (11,22,...,99), yes rate on non-example repdigit targets minus yes rate on other non-example targets; observed > null means the model lacks a "repeated digits" concept people use. (observed 0.0421 vs null mean -0.396, z=2.47, p=0.04, q=0.107)

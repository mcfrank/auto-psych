# Critique of `progression_graded_magnitude_band`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **first_example_digit_lift** — Among multi-example sets without a shared final digit, yes-rate on non-example targets sharing the FIRST example's final digit (but not the last's) minus yes-rate on targets sharing no example's final digit; the model gives only the last example a lingering digit, so observed above null means a primacy digit effect the model misses. (observed 0.143 vs null mean 0.065, z=2.72, p=0.00799, q=0.0639)
- **single_set_breadth_dispersion** — Standard deviation across single-example sets (with >=20 responses) of the set-level yes-rate on non-example targets; observed above null means people's generalization breadth varies more from seed to seed (some single numbers suggest broad, others narrow concepts) than the model produces, below means the model makes seeds too different. (observed 0.125 vs null mean 0.0912, z=2.26, p=0.034, q=0.136)

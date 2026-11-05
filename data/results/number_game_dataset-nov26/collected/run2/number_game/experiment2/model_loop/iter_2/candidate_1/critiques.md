# Critique of `gap_step_progression_overlap`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_yes_gap** — Yes-rate on single-example sets minus yes-rate on multi-example sets (non-example targets only); observed above null means the model under-generalizes from one example (too narrow/conservative on n=1), below means it over-generalizes. (observed 0.147 vs null mean 0.091, z=2.90, p=0.004, q=0.032) [survives FDR]
- **single_example_proximity_gap** — In single-example sets, yes-rate on targets within 15 of the example minus yes-rate on targets more than 40 away; the model has no numeric-distance term for n=1 (only one-sided bounds), so observed above null means people generalize from a single example by magnitude similarity the model misses, below means they ignore magnitude more than modelled. (observed 0.0556 vs null mean -0.0331, z=2.58, p=0.00799, q=0.032) [survives FDR]
- **first_example_digit_lift** — Among multi-example sets without a shared final digit, yes-rate on non-example targets sharing the FIRST example's final digit (but not the last's) minus yes-rate on targets sharing no example's final digit; the model gives only the last example a lingering digit, so observed above null means a primacy digit effect the model misses. (observed 0.143 vs null mean 0.0717, z=2.51, p=0.012, q=0.032) [survives FDR]

# Critique of `bigelow_piantadosi_independent`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **outside_range_yes_multi** — Yes rate for targets outside [min, max] of the examples on sets with 3-4 examples; observed above null means the model is too confident that rich example sets exclude out-of-range numbers (under-generalizes beyond the range), below means it over-generalizes. (observed 0.389 vs null mean 0.489, z=-3.80, p=0.002, q=0.00799) [survives FDR]
- **within_cell_response_variability** — Mean over (set, target) cells of p(1-p), p = cell yes rate across repeated presentations; observed above null means the model's predicted probabilities are too extreme (near 0/1) relative to graded human responding, below means too graded. (observed 0.192 vs null mean 0.178, z=3.40, p=0.002, q=0.00799) [survives FDR]

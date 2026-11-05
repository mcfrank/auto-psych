# Critique of `range_size_prior_concepts`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_multiple_targets_rate** — Yes rate on one-example sets (example > 1) for non-example targets that are multiples of the example (10->80, 11->55, 6->18); observed above null means the model still under-generalizes a lone example to its multiples rule (below = over-generalizes). (observed 0.725 vs null mean 0.439, z=7.11, p=0.002, q=0.016) [survives FDR]
- **in_minus_out_range_3plus** — For sets with 3+ examples, non-example-target yes rate inside [min,max] of the examples minus outside it; observed above null means people confine generalization to the examples' range more than the model (below = the model over-confines, e.g. via its range-size prior). (observed 0.0587 vs null mean 0.0108, z=2.68, p=0.00799, q=0.0266) [survives FDR]
- **single_example_nonmultiple_targets_rate** — Yes rate on one-example sets for non-example targets that are NOT multiples of the example (6->8, 14->74, 100->20, 1->27 count as non-multiples except x=1); observed above null means people generalize a lone example more diffusely than the model (below = model too diffuse beyond the multiples rule). (observed 0.455 vs null mean 0.364, z=2.62, p=0.00999, q=0.0266) [survives FDR]
- **within_set_target_sd** — Mean over sets with >=5 distinct targets of the SD of per-target yes rates within the set; observed above null means people discriminate between targets of the same set more sharply than the model (below = model's gradients are too sharp / all-or-none). (observed 0.16 vs null mean 0.175, z=-2.24, p=0.03, q=0.0599)

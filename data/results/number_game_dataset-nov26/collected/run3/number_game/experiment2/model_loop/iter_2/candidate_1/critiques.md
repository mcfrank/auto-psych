# Critique of `range_prior_sample_size_neglect`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_multiple_targets_rate** — Yes rate on one-example sets (example > 1) for non-example targets that are multiples of the example (6->18, 11->55, 25->100); observed above null means the neglect model still under-generalizes a lone example to its multiples rule (below = over-generalizes). (observed 0.725 vs null mean 0.503, z=4.71, p=0.002, q=0.00799) [survives FDR]
- **in_minus_out_range_3plus** — For sets with 3+ examples, non-example-target yes rate inside [min,max] of the examples minus outside it; observed above null means people confine generalization to the examples' range more than the model (below = model over-confines). (observed 0.0587 vs null mean 0.00423, z=3.11, p=0.002, q=0.00799) [survives FDR]
- **high_minus_low_target_rate** — Across all sets, non-example-target yes rate for targets > 50 minus targets <= 50; observed above null means people favour large numbers more than the model (a magnitude bias the model lacks); below = model over-favours large targets. (observed 0.0492 vs null mean 0.0135, z=2.36, p=0.026, q=0.0693)

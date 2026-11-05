# Critique of `digit_rules_need_comparison`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_digit_sets_matching_outside_range** — For 3+-example sets whose examples all share a last digit, yes rate on non-example targets with that same last digit lying outside [min,max] (48,78,38,98->8,18; 64,94,84->14); observed above null means people extend a digit rule beyond the examples' range more than the model's range-bounded rules allow (below = model over-extends). (observed 0.838 vs null mean 0.935, z=-3.95, p=0.002, q=0.00799) [survives FDR]
- **single_example_multiples_rate** — Yes rate on one-example sets (example > 1) for non-example targets that are multiples of the example (6->18, 11->55, 25->100); observed above null means the model, even with digit rules discounted, still under-generalizes a lone example to its multiples (below = over-generalizes). (observed 0.725 vs null mean 0.567, z=3.33, p=0.002, q=0.00799) [survives FDR]
- **high_minus_low_multi_example** — For sets with 2+ examples, non-example-target yes rate for targets > 50 minus targets <= 50; observed above null means people favour large targets more than the model (a magnitude bias the model lacks), below = model over-favours large targets. (observed 0.0455 vs null mean 0.0132, z=2.15, p=0.036, q=0.0959)
- **in_minus_out_range_two_examples** — For two-example sets, non-example-target yes rate for targets strictly inside [min,max] of the examples minus targets outside it; observed above null means people confine generalization to the examples' range more than the model does for pairs (below = model over-confines). (observed 0.0781 vs null mean 0.0108, z=2.04, p=0.048, q=0.0959)

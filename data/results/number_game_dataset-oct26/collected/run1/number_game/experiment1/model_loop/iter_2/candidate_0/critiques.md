# Critique of `shared_property_rule_matching`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **set_level_yes_rate_variance** — Variance across number sets of the per-set mean yes rate (sets with at least 9 responses); observed above null means sets differ in overall generalization breadth more than the single-intercept rule model allows (model too uniform across sets). (observed 0.0468 vs null mean 0.0278, z=4.30, p=0.004, q=0.016) [survives FDR]
- **inside_minus_outside_range** — Among sets with 2-4 examples (target not itself an example), yes rate for targets strictly inside the examples' min-max range minus targets outside it; observed above null means the model misses interval/range-based generalization. (observed 0.00663 vs null mean -0.0829, z=3.91, p=0.002, q=0.016) [survives FDR]
- **many_minus_single_example_yes_rate** — Yes rate on sets with 3-4 examples minus yes rate on 1-example sets (targets not in the set); observed below null means people tighten generalization with more examples (size principle) more than the rule model predicts, above null means the model over-tightens. (observed -0.0539 vs null mean 0.00502, z=-2.13, p=0.038, q=0.101)

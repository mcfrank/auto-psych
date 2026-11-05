# Critique of `span_interval_similarity`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_vs_multi_yes_gap** — Yes rate on one-example sets minus yes rate on 2-4-example sets; observed > null means the model under-produces the extra breadth of generalization people show from a single example (or over-produces it if observed < null). (observed 0.18 vs null mean 0.125, z=3.45, p=0.002, q=0.016) [survives FDR]
- **single_example_distance_gradient** — For one-example sets, yes rate for targets within 10 of the example minus yes rate for targets more than 30 away; observed > null means people's magnitude generalization from a lone example is steeper than the model's, observed < null shallower. (observed 0.0112 vs null mean -0.0757, z=2.99, p=0.00599, q=0.024) [survives FDR]
- **item_mean_variance** — Variance across (set, target) pairs of the pair's mean yes rate; observed > null means people differentiate items more sharply (more all-yes / all-no pairs) than the model, observed < null means the model is over-confident. (observed 0.0601 vs null mean 0.0536, z=2.72, p=0.00999, q=0.0266) [survives FDR]

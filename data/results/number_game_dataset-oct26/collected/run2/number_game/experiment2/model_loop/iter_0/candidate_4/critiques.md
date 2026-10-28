# Critique of `dual_route_shared_factor_rule_exemplar`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_rating_correlation** — Pearson correlation between trial_index and rating; observed above the null means people drift toward saying yes more over the session, a time trend the stationary model does not produce (below means drift toward no). (observed 0.0511 vs null mean 0.00899, z=3.14, p=0.004, q=0.016) [survives FDR]
- **shared_last_digit_contrast** — In sets of 2+ examples that all end in the same digit, yes-rate for (non-identical) targets ending in that digit minus targets ending in another digit; observed above the null means people apply an explicit "ends in d" rule more sharply than the model's final-digit similarity dimension predicts, below means less. (observed 0.603 vs null mean 0.471, z=2.91, p=0.004, q=0.016) [survives FDR]
- **cell_response_dispersion** — Mean over (number_set, target) cells of the within-cell Bernoulli variance p(1-p) of the repeated responses; observed below the null means people agree with each other on each item more (responses more deterministic/extreme) than the model's predicted probabilities imply, above means the model is overconfident. (observed 0.182 vs null mean 0.189, z=-2.40, p=0.022, q=0.0586)

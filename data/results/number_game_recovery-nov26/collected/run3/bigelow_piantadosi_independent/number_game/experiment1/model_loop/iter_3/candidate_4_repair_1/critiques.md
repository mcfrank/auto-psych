# Critique of `erlang_interval_simplicity_search`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_divisor_rule_yes_rate** — For 2+ example sets whose examples are all multiples of some m in 3..10, yes rate for non-example targets that are multiples of such a shared m; observed < null means people extend arithmetic (multiple-of-m) rules less than the model, observed > null means more. (observed 0.714 vs null mean 0.604, z=4.11, p=0.002, q=0.016) [survives FDR]
- **very_near_minus_moderate_near** — Yes rate for non-example targets 1-3 from the nearest example minus yes rate for targets 4-10 away; the model's intervals imply a monotone decline (positive value), so observed < null means people generalise relatively less to immediate neighbours than to moderately near numbers, observed > null means a steeper local gradient. (observed -0.116 vs null mean -0.0652, z=-2.16, p=0.032, q=0.101)
- **recency_nearest_example** — For 2+ example sets and non-example targets, yes rate when the nearest example is the last-shown one minus yes rate when the nearest is the first-shown one; the model is order-invariant, so observed < null means people generalise less around the most recent example (primacy), observed > null means a recency advantage. (observed -0.0673 vs null mean -0.0331, z=-2.13, p=0.038, q=0.101)

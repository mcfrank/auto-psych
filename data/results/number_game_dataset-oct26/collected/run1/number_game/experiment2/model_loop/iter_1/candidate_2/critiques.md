# Critique of `nearest_decade_digit_common_factor`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_yes_slope** — OLS slope of rating on trial_index scaled to change in yes-probability per 100 trials; t_observed differing from null_mean (~0) means participants drift toward more (positive) or fewer (negative) yes responses over the session, which the stationary model cannot produce. (observed 0.0186 vs null mean 0.00298, z=3.09, p=0.002, q=0.016) [survives FDR]
- **item_yes_rate_dispersion** — Standard deviation across (set, target) items of the item-level yes rate; t_observed > null_mean means human responses are more all-or-none / extreme across items than the model's graded predictions (model under-produces item polarization). (observed 0.252 vs null mean 0.233, z=3.02, p=0.004, q=0.016) [survives FDR]
- **cross_decade_near_magnitude_yes_rate** — Yes rate for targets within 1-4 of some example numerically but in a different decade and sharing no final digit with any example (e.g. 29 vs 31); t_observed > null_mean means the model under-produces magnitude-proximity generalization that ignores decade boundaries. (observed 0.296 vs null mean 0.393, z=-2.87, p=0.00799, q=0.0213) [survives FDR]
- **inside_vs_outside_range_yes_gap** — In multi-example sets spanning >=20, among targets sharing no final digit with any example and not an example, yes rate for targets strictly inside [min,max] of the examples minus targets outside it; t_observed > null_mean means the model under-produces interval (range-interpolation) generalization relative to extrapolation. (observed 0.0801 vs null mean -0.00482, z=2.46, p=0.012, q=0.024) [survives FDR]

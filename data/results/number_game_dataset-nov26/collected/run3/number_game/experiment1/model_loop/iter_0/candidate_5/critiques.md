# Critique of `bigelow_piantadosi_mixture`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **min_rule_member_yes_rate** — Yes-rate on targets inside the smallest of the 45 rules consistent with all examples; observed below null means the model over-endorses minimal-rule members (humans are less confident in the winning rule). (observed 0.482 vs null mean 0.573, z=-7.70, p=0.002, q=0.016) [survives FDR]
- **member_gradedness_spread** — Standard deviation, across (set,target) cells whose target is in the minimal consistent rule, of the cell mean rating after subtracting the set's member mean; observed above null means humans grade rule members unequally (typicality), while the model treats all members of a concept alike. (observed 0.18 vs null mean 0.152, z=3.08, p=0.00599, q=0.024) [survives FDR]
- **shared_last_digit_nonmember_yes_rate** — Yes-rate on targets outside the minimal consistent rule that share their last digit with at least half of the examples, minus yes-rate on other such non-member targets; observed above null means the model misses partial featural (last-digit) similarity generalization when no digit rule fits all examples. (observed 0.414 vs null mean 0.291, z=2.39, p=0.028, q=0.0559)
- **nonmember_proximity_slope** — Among targets outside the minimal consistent rule, the OLS slope of rating on log(1 + distance to nearest example); observed more negative than null means humans show stronger numeric-proximity generalization than the model's intervals produce. (observed 0.0327 vs null mean -0.00286, z=2.29, p=0.028, q=0.0559)

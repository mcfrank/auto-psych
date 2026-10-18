# Critique of `exemplar_feature_similarity_gcm`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **item_level_rating_variance** — Variance across (set, target) items of the item mean yes rate; observed above null means people's judgments are more sharply all-or-none across items than the model's graded similarity predicts (model too flat), below means the model over-discriminates. (observed 0.0551 vs null mean 0.0326, z=8.10, p=0.002, q=0.004) [survives FDR]
- **rule_consistent_minus_inconsistent** — Among multi-example sets whose examples all share at least one arithmetic property (even/odd, mult of 3/4/5/10, same final digit, prime, square, power of 2), yes rate for targets having ALL shared properties minus yes rate for targets lacking any; observed above null means people apply the shared rule more sharply than graded summed similarity predicts. (observed 0.254 vs null mean 0.0989, z=5.30, p=0.002, q=0.004) [survives FDR]
- **singleton_minus_multi_yes_rate** — Yes rate on 1-example sets minus yes rate on sets with 2-4 examples; summed similarity predicts more examples -> more yes (negative value), so an observed value above the null means the model under-produces generalization from single examples relative to multi-example sets (no size-principle tightening). (observed 0.0597 vs null mean -0.0547, z=4.08, p=0.002, q=0.004) [survives FDR]
- **far_target_yes_rate** — Yes rate for targets more than 25 away from every example (magnitude-distant targets); observed above null means the model under-predicts acceptance of numerically distant targets (people generalize by rule, not by magnitude proximity), below means it over-predicts. (observed 0.512 vs null mean 0.443, z=3.61, p=0.002, q=0.004) [survives FDR]

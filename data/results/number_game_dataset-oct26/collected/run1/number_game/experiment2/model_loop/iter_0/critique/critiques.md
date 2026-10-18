# Critique of `digit_similarity_with_common_factor`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **range_inclusion_gap** — Among multi-example sets and targets not in the set, yes-rate for targets inside [min,max] of the examples minus yes-rate for targets outside; observed above model mean means people use numeric range/magnitude that the digit model ignores. (observed 0.0552 vs null mean -0.0545, z=6.42, p=0.002, q=0.00799) [survives FDR]
- **same_decade_boost_no_digit_match** — Among targets not sharing a final digit with any example (and not in the set), yes-rate for targets sharing a tens digit (decade) with some example minus others; observed above model mean means the model misses leading-digit/decade similarity. (observed -0.0245 vs null mean -0.114, z=4.24, p=0.002, q=0.00799) [survives FDR]
- **item_rate_variance** — Variance across (set,target) items of the per-item yes-rate; observed above model mean means items differ more systematically than the model's three features can explain (unexplained stimulus structure). (observed 0.0636 vs null mean 0.0527, z=3.67, p=0.004, q=0.0107) [survives FDR]
- **single_vs_multi_example_yes_gap** — Overall yes-rate on single-example sets minus yes-rate on 3-4-example sets (targets not in set); observed differing from model mean means the model mis-captures how generalization breadth changes with number of examples (size principle). (observed 0.00424 vs null mean 0.044, z=-2.38, p=0.014, q=0.028) [survives FDR]
- **yes_rate_target_in_set** — Yes-rate among trials whose target is literally one of the examples; observed above model mean means the model under-produces near-certain yes for shown examples. (observed 0.798 vs null mean 0.744, z=2.36, p=0.024, q=0.0384) [survives FDR]

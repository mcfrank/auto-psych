# Critique of `person_switch_ideal_pattern_rule_built`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **ending_alternation_equal_switch** — Among pairs with equal numbers of switches where exactly one sequence ends with a switch (last two flips differ), the rate of choosing the sequence that ends with a switch; observed above null means people weigh the ending (recency, avoiding a terminal repeat) more than the model, which is blind to switch position, below means the reverse. (observed 0.521 vs null mean 0.458, z=2.97, p=0.00999, q=0.0799)
- **run_length_variety_equal_switch** — Among pairs with equal numbers of switches whose sequences differ in the number of distinct run lengths, the rate of choosing the sequence with more distinct run lengths (mixed 1s, 2s, 3s); observed above null means people prize irregular run lengths more than the model predicts (it under-produces this choice), below means less. (observed 0.518 vs null mean 0.453, z=2.31, p=0.036, q=0.144)

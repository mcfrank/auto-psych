# Critique of `iter2_candidate3`

4 of 7 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **perfect_alternator_choice_rate** — Among pairs where exactly one sequence alternates on every flip (e.g. HTHTHTHT), the rate of choosing the perfect alternator; observed below null means people reject too-regular alternation that the model linearly rewards (model over-produces choosing perfect alternation), above null the reverse. (observed 0.532 vs null mean 0.598, z=-4.22, p=0.002, q=0.00699) [survives FDR]
- **more_distinct_triplets_at_equal_switches** — Among pairs with equal switch counts but different numbers of distinct length-3 substrings, the rate of choosing the sequence with more distinct triplets; observed above null means people reward local pattern variety that span and switching miss (model under-produces it). (observed 0.634 vs null mean 0.571, z=3.52, p=0.002, q=0.00699) [survives FDR]
- **more_switches_among_high_switch_pairs** — Among pairs where both sequences switch on at least 60% of transitions and switch rates differ, the rate of choosing the more-switching sequence; observed below null means the alternation preference saturates/inverts at high switch rates (inverted-U) while the linear model over-produces it. (observed 0.525 vs null mean 0.465, z=2.57, p=0.00799, q=0.0186) [survives FDR]
- **shorter_longest_run_at_equal_switches** — Among pairs with equal H/T switch counts but different longest-run lengths, the rate of choosing the sequence with the shorter longest run; observed above null means people avoid long streaks more than the span+switching model predicts (model under-produces streak avoidance). (observed 0.625 vs null mean 0.585, z=2.18, p=0.03, q=0.0524)

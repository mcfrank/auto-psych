# Critique of `balance_aware_heads_alternation_ideal`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shorter_longest_run_at_equal_switches** — Among pairs with equal switch counts but different longest-run lengths, the rate of choosing the sequence with the shorter longest run; observed above null means people penalise long streaks beyond what switching rate predicts (model under-produces streak avoidance). (observed 0.625 vs null mean 0.535, z=4.75, p=0.002, q=0.00799) [survives FDR]
- **more_distinct_triplets_at_equal_switches** — Among pairs with equal switch counts but different numbers of distinct length-3 substrings, the rate of choosing the sequence with more distinct triplets (richer local patterns); observed above null means people reward local pattern variety that the alternation/periodicity/balance terms miss. (observed 0.634 vs null mean 0.554, z=4.21, p=0.002, q=0.00799) [survives FDR]
- **participant_heads_preference_variance** — Variance across participants of their rate of choosing the more-heads sequence (pairs differing in #H); observed above null means the heads preference differs between people, which the model's single shared heads weight under-produces. (observed 0.00495 vs null mean 0.00335, z=2.94, p=0.018, q=0.048) [survives FDR]
- **participant_side_bias_variance** — Variance across participants of their left-choice rate; observed above null means individuals have stable left/right response biases the model (no side bias) does not produce. (observed 0.00514 vs null mean 0.00378, z=2.44, p=0.032, q=0.0511)
- **pair_consensus_extremity_long** — Mean over length-8 pairs of |proportion choosing left - 0.5|; observed above null means people agree on long-sequence pairs more strongly than the model predicts (model too noisy/under-confident there), below null means the model is over-confident. (observed 0.231 vs null mean 0.217, z=2.09, p=0.03, q=0.0511)

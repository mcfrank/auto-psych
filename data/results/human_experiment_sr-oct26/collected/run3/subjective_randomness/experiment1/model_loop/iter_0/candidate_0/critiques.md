# Critique of `local_representativeness`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_majority_agreement_sd** — Across-participant SD of each person's rate of agreeing with the item's majority choice; observed > null means people differ in how they judge randomness more than the single pooled model (one shared beta and weights) produces. (observed 0.149 vs null mean 0.0498, z=17.35, p=0.002, q=0.00799) [survives FDR]
- **ends_with_repeat_choice** — Among pairs where exactly one sequence ends with a repeat (last two flips equal), the rate of choosing that sequence; observed < null means people penalise a terminal repetition (recency/gambler's-fallacy salience) more than the model's position-blind features predict. (observed 0.183 vs null mean 0.247, z=-4.72, p=0.002, q=0.00799) [survives FDR]
- **shorter_longest_run_choice** — Among pairs whose longest runs differ, the rate of choosing the sequence with the shorter longest run; observed > null means people penalise long streaks more than the balance/alternation features predict (observed < null: less). (observed 0.831 vs null mean 0.795, z=3.34, p=0.004, q=0.0107) [survives FDR]
- **balanced_pairs_more_alternation_choice** — Among pairs whose head counts differ by at most one but different numbers of alternations, the rate of choosing the sequence with more alternations; observed > null means people favour alternation (when balance is nearly tied) more than the model's alternation term predicts. (observed 0.703 vs null mean 0.656, z=2.63, p=0.014, q=0.028) [survives FDR]
- **participant_left_rate_sd** — Across-participant SD of each person's left-choice rate; observed > null means individual side biases vary more than the model's single shared side_bias allows. (observed 0.0778 vs null mean 0.0624, z=2.30, p=0.024, q=0.0384) [survives FDR]

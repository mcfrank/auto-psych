# Critique of `lapse_individual_balance_alternation`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **end_streak_pref_matched_alt** — Among pairs with alternation rates within 0.15 where exactly one sequence starts or ends with a run of 3+ identical flips, the proportion choosing the sequence WITHOUT an edge streak; observed above null means streaks at the sequence edges are penalised beyond the model's alternation/balance terms. (observed 0.608 vs null mean 0.529, z=2.41, p=0.03, q=0.12)
- **participant_sd_maxrun_pref** — Across participants, the SD of each person's rate of choosing the sequence with the shorter longest run (pairs differing in longest run); observed above null means individuals differ in streak aversion more than the model's balance/alternation/lapse heterogeneity produces. (observed 0.242 vs null mean 0.221, z=2.05, p=0.03, q=0.12)
- **left_choice_rate** — Overall proportion of trials choosing the left sequence; observed away from the null (~0.5) means a side bias the model (which has no side term) fails to produce. (observed 0.525 vs null mean 0.511, z=2.02, p=0.05, q=0.133)

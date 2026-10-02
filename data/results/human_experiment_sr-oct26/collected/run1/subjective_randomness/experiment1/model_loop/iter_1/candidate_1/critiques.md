# Critique of `personal_ideal_alternation`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **balance_pref_matched_alternation** — Among pairs of irregular sequences (both alternation rates >= 0.5) whose alternation rates differ by at most 1/7 but whose H/T imbalance differs, the proportion choosing the more balanced sequence; observed above the null means the model under-predicts a preference for balanced H/T counts that alternation alone cannot capture. (observed 0.581 vs null mean 0.5, z=4.04, p=0.002, q=0.00799) [survives FDR]
- **maxrun_effect_beyond_alternation** — Among pairs where both sequences have alternation rate >= 0.5, the coefficient on (longest run of right minus longest run of left) in a linear-probability regression of chose_left on [1, alt_left - alt_right, maxrun_right - maxrun_left]; observed above the null means people penalise long runs beyond what the alternation rate captures, which the model under-predicts. (observed 0.0794 vs null mean 0.0112, z=3.16, p=0.002, q=0.00799) [survives FDR]

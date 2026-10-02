# Critique of `ideal_alternation_with_balance`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_spread_balance_pref** — Standard deviation across participants of each participant's proportion choosing the more H/T-balanced sequence among pairs whose imbalance differs by at least 0.25; observed above the null means people differ in how much they weight balance more than the model's single shared balance weight allows. (observed 0.18 vs null mean 0.143, z=4.40, p=0.002, q=0.016) [survives FDR]
- **short_seq_alt_agreement** — Among trials with sequence length <= 5 whose alternation rates differ, the proportion choosing the higher-alternation sequence minus the same proportion among length-8 trials; observed above the null means short sequences are judged by alternation more (or long ones less) than the model's length-independent sensitivity predicts, below means the reverse. (observed 0.116 vs null mean 0.168, z=-2.11, p=0.046, q=0.136)

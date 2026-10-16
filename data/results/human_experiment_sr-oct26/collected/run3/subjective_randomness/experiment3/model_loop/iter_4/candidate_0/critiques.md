# Critique of `streak_weary_chance_pseudoflip`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **switches_vs_balance_conflict** — Among length-8 pairs where the more-switching sequence has the more lopsided heads/tails count (|#H - #T| strictly larger), the rate of choosing the more-switching sequence; observed below null means people weigh heads/tails balance against switching more than the model predicts, above means less. (observed 0.609 vs null mean 0.575, z=2.14, p=0.044, q=0.176)
- **low_switch_pairs_more_switches** — Among pairs of length >= 6 where both sequences have at most 2 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed above null means people still reward the first breaks of a long streak more than the gambler's-run model predicts, below means the run-length term now over-rewards them. (observed 0.872 vs null mean 0.848, z=2.11, p=0.042, q=0.176)

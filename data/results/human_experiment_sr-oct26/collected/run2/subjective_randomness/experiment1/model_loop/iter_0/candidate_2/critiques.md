# Critique of `local_representativeness`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_alternation_pref_sd** — Across-participant SD of each person's rate of choosing the more-alternating sequence (trials whose alternation counts differ); observed > null means the single-population model under-produces individual differences in alternation preference. (observed 0.16 vs null mean 0.0535, z=17.13, p=0.002, q=0.00799) [survives FDR]
- **high_alternation_preference** — Among trials where both sequences have alternation proportion >= 0.5 and differ in alternation count, the rate of choosing the more-alternating one; observed > null means humans prefer heavier alternation than the model's fitted alternation prototype allows, observed < null means the model over-rewards alternation at the high end. (observed 0.573 vs null mean 0.503, z=2.80, p=0.002, q=0.00799) [survives FDR]
- **alternation_pref_drift_over_trials** — Rate of choosing the more-alternating sequence in the second half of each session minus the first half; positive observed vs null means preferences drift toward alternation over the session, which the static model cannot produce. (observed -0.0196 vs null mean 0.0207, z=-2.34, p=0.022, q=0.0559)
- **pair_choice_extremity** — Mean over distinct (unordered) pairs of |share choosing the lexicographically smaller sequence - 0.5|; observed > null means the model's pair predictions are too weak (humans more unanimous), observed < null means the model is overconfident. (observed 0.251 vs null mean 0.229, z=2.24, p=0.028, q=0.0559)

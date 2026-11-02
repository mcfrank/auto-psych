# Critique of `noisy_examples_erlang_intervals`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **new_set_first_trial_effect** — Yes-rate on trials whose number_set differs from the set on the preceding trial_index (first query after a set change) minus yes-rate on trials repeating the previous trial's set; the model is history-free, so observed below null means people are more conservative right after a new set appears (and loosen with repeated queries), above null the reverse. (observed -0.0702 vs null mean -0.0254, z=-2.29, p=0.026, q=0.208)

# Critique of `erlang_width_compressed_intervals`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_last_digit_yes** — Yes rate for non-example targets whose last digit matches the shared last digit of a >= 2-example set (all examples end in the same digit); observed > null means people endorse "ends in d" rule extensions more than the model, observed < null less. (observed 0.978 vs null mean 0.833, z=3.27, p=0.002, q=0.016) [survives FDR]
- **dist2_minus_dist1_yes** — Yes rate for targets exactly 2 from the nearest example minus yes rate for targets exactly 1 away; observed > null means people favour same-parity neighbours over adjacent numbers more than the model's interval/parity mix predicts. (observed 0.25 vs null mean 0.162, z=2.86, p=0.004, q=0.016) [survives FDR]
- **opposite_parity_yes_in_parity_sets** — Yes rate for targets of the opposite parity to the examples, among sets of >= 2 examples all sharing one parity; observed < null means the model's interval/noise mass over-predicts yes to parity-violating targets, observed > null means it under-predicts. (observed 0.151 vs null mean 0.173, z=-2.61, p=0.00599, q=0.016) [survives FDR]
- **lag1_response_autocorr** — Correlation between consecutive responses within a participant (sessions split where trial_index resets, rows ordered by trial_index); observed > null means responses are serially perseverative beyond what the trial-independent model produces. (observed 0.137 vs null mean 0.101, z=2.41, p=0.016, q=0.032) [survives FDR]
- **early_trial_yes_excess** — Yes rate on trials with trial_index < 90 minus yes rate on later trials; observed > null means people say yes more early in the session (a drift the stationary model cannot produce). (observed 0.148 vs null mean 0.18, z=-2.05, p=0.048, q=0.0767)

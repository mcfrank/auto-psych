# Critique of `compressed_rule_extension`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **opp_parity_just_outside_yes** — In sets of >=2 whose examples all share a parity, yes-rate for opposite-parity targets 1-3 outside the example span; observed below null means the model over-extends ranges to opposite-parity neighbours (people extrapolate by parity), above null means people extend magnitude ranges past the edge more than the model. (observed 0.444 vs null mean 0.523, z=-2.66, p=0.012, q=0.0559)
- **parity_gap_inside_minus_outside** — In sets of >=2 sharing a parity, (same-minus-opposite parity yes gap for non-example targets inside the span) minus (the same gap for targets outside the span); observed below null means people interpolate by magnitude but extrapolate by parity more than the model, above null means parity matters more inside the span than the model predicts. (observed -0.0522 vs null mean 0.0269, z=-2.48, p=0.014, q=0.0559)
- **set_size_near_outside_gap** — Yes-rate for non-example targets 1-10 outside the span in sets of 3-4 examples minus the same in sets of 2; observed below null means more examples tighten generalization more than the model's size principle predicts, above null means people tighten less with more examples. (observed 0.194 vs null mean 0.138, z=2.00, p=0.046, q=0.0999)
- **below_asym_min_first** — In sets of >=2, yes-rate for targets 1-10 below the min when the smallest example is shown first minus when it is not; the model has no first-shown cue, so observed below null means a first-shown minimum marks the range bottom (restricts downward extension), above null means it invites downward extension. (observed 0.0935 vs null mean 0.171, z=-1.99, p=0.05, q=0.0999)

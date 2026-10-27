# run3 (mild_symmetric_trick_coin_streak_weary): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `r3_full` | 14 + 4p | -10908.3 | trick_coin_spread | 2.3 ± 11.6 | 0.20 | drop trick_coin_spread |
| 2 | `r3_minus_spread` | 13 + 4p | -10910.6 | length_padding | 2.3 ± 1.5 | 1.51 | drop length_padding |
| 3 | `r3_minus_spread_pad` | 12 + 4p | -10912.9 | gamblers_run | 2.6 ± 5.4 | 0.48 | drop gamblers_run |
| 4 | `r3_minus_spread_pad_gr` | 11 + 4p | -10915.5 | second_order | 8.6 ± 8.6 | 1.00 | drop second_order |
| 5 | `r3_minus_spread_pad_so_gr` | 10 + 4p | -10924.1 | side_personal | 13.2 ± 4.4 | 2.98 | stop: the cheapest removal (side_personal) costs 13.2 nats, 2.98 SE >= 2.0 |

**Lean model:** `r3_minus_spread_pad_so_gr`, 10 population parameters + 4 person-level vectors. Kept: markov, trick_coin, motif, motif_weight, motif_weight_personal, beta_personal, switch_belief, switch_belief_personal, length_scaling, side_bias, side_personal. Removed: trick_coin_spread, length_padding, second_order, gamblers_run. Against the starting model (the full final): 15.9 ± 14.5 nats of ELPD-CV.

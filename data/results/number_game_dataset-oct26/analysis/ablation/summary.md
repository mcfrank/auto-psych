# ng (even_rule_digit_size_principle_counting): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `ng_full` | 14 + 0p | -4696.2 | parity_preempt | 1.7 ± 2.5 | 0.67 | drop parity_preempt |
| 2 | `ng_minus_parpre` | 13 + 0p | -4697.8 | count_step_cost | 3.1 ± 5.5 | 0.55 | drop count_step_cost |
| 3 | `ng_minus_parpre_step` | 12 + 0p | -4700.9 | size_principle | 2.2 ± 6.6 | 0.33 | drop size_principle |
| 4 | `ng_minus_parpre_step_size` | 11 + 0p | -4703.1 | even_rule | 10.8 ± 7.4 | 1.48 | drop even_rule |
| 5 | `ng_minus_parpre_step_size_even` | 10 + 0p | -4713.9 | counting | 11.8 ± 6.8 | 1.74 | drop counting |
| 6 | `ng_minus_parpre_count_step_size_even` | 9 + 0p | -4725.8 | mutual_standard | 30.0 ± 18.5 | 1.62 | drop mutual_standard |
| 7 | `ng_minus_parpre_std_count_step_size_even` | 8 + 0p | -4755.7 | magnitude_preempt | 51.1 ± 19.6 | 2.61 | stop: the cheapest removal (magnitude_preempt) costs 51.1 nats, 2.61 SE >= 2.0 |

**Lean model:** `ng_minus_parpre_std_count_step_size_even`, 8 population parameters + 0 person-level vectors. Kept: intercept, similarity, parity, common_factor, round_factor, decade_band, magnitude_preempt. Removed: parity_preempt, mutual_standard, counting, count_step_cost, size_principle, even_rule. Against the starting model (the full final): 59.6 ± 20.4 nats of ELPD-CV.

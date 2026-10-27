# run2 (edge_streak_primacy_recency_pattern_rule): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `r2_full` | 15 + 6p | -10572.7 | final_run | -1.4 ± 3.6 | -0.39 | drop final_run |
| 2 | `r2_minus_fin` | 14 + 6p | -10571.3 | opening_run | 0.8 ± 1.1 | 0.79 | drop opening_run |
| 3 | `r2_minus_fin_open` | 13 + 6p | -10572.1 | side_personal | 13.0 ± 4.6 | 2.84 | stop: the cheapest removal (side_personal) costs 13.0 nats, 2.84 SE >= 2.0 |

**Lean model:** `r2_minus_fin_open`, 13 population parameters + 6 person-level vectors. Kept: span_ideal_personal, span_beta_personal, length_scaling, switch, switch_ideal_personal, switch_weight_personal, variety, rule_built, pattern_gain_personal, side_personal. Removed: final_run, opening_run. Against the starting model (the full final): -0.6 ± 3.6 nats of ELPD-CV.

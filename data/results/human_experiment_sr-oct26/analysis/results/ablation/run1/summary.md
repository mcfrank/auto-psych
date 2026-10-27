# run1 (personal_pattern_detection_gain): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `r1_full` | 17 + 6p | -10362.2 | final_run | -1.4 ± 2.7 | -0.53 | drop final_run |
| 2 | `r1_minus_fin` | 16 + 6p | -10360.8 | asymmetry | 15.1 ± 13.2 | 1.15 | drop asymmetry |
| 3 | `r1_minus_asym_fin` | 15 + 6p | -10375.9 | antisymmetry | 37.3 ± 15.9 | 2.34 | stop: the cheapest removal (antisymmetry) costs 37.3 nats, 2.34 SE >= 2.0 |

**Lean model:** `r1_minus_asym_fin`, 15 population parameters + 6 person-level vectors. Kept: ideal_personal, beta_personal, imbalance, imbalance_personal, longest_run, longest_run_personal, palindrome, variety, lopsided, antisymmetry, gain_personal, lapse, lapse_personal. Removed: asymmetry, final_run. Against the starting model (the full final): 13.7 ± 13.6 nats of ELPD-CV.

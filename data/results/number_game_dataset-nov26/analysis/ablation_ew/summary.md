# ew (exception_weighted_rule_rejection): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `ew_full` | 56 + 0p | -13024.1 | exception | -28.3 ± 19.6 | -1.44 | drop exception |
| 2 | `ew_minus_exc` | 55 + 0p | -12995.8 | size_prior | -4.9 ± 4.6 | -1.05 | drop size_prior |
| 3 | `ew_minus_exc_size` | 54 + 0p | -12990.9 | zero_split | 0.3 ± 0.5 | 0.59 | drop zero_split |
| 4 | `ew_minus_exc_zero_size` | 53 + 0p | -12991.2 | parity_concepts | -0.4 ± 1.5 | -0.25 | drop parity_concepts |
| 5 | `ew_minus_exc_zero_par_size` | 52 + 0p | -12990.9 | range_evidence | 9.6 ± 10.6 | 0.90 | drop range_evidence |
| 6 | `ew_minus_exc_zero_rev_par_size` | 51 + 0p | -13000.4 | explaining_away | 27.0 ± 31.2 | 0.87 | drop explaining_away |
| 7 | `ew_minus_exc_zero_rev_par_size_pre` | 50 + 0p | -13027.5 | bounded_rules | 50.2 ± 26.2 | 1.92 | drop bounded_rules |
| 8 | `ew_minus_exc_zero_rev_par_bnd_size_pre` | 49 + 0p | -13077.7 | below_threshold | 55.0 ± 23.1 | 2.38 | stop: the cheapest removal (below_threshold) costs 55.0 nats, 2.38 SE >= 2.0 |

**Lean model:** `ew_minus_exc_zero_rev_par_bnd_size_pre`, 49 population parameters + 0 person-level vectors. Kept: rules_vs_ranges, rule_prior, tempering, neglect, above_threshold, below_threshold, lone_example_discount. Removed: exception, zero_split, range_evidence, parity_concepts, bounded_rules, size_prior, explaining_away. Against the starting model (the full final): 53.6 ± 49.1 nats of ELPD-CV.

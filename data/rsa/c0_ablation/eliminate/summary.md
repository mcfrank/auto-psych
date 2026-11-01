# c0 (boundary_avoidable_ambiguity_listener): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `c0_full` | 17 + 0p | -35374.2 | distinct | -0.4 ± 2.0 | -0.22 | drop distinct |
| 2 | `c0_minus_dist` | 16 + 0p | -35373.8 | color | -0.0 ± 0.3 | -0.09 | drop color |
| 3 | `c0_minus_dist_col` | 15 + 0p | -35373.8 | popout | 0.0 ± 0.2 | 0.17 | drop popout |
| 4 | `c0_minus_dist_col_pop` | 14 + 0p | -35373.8 | valence | 0.2 ± 0.2 | 1.04 | drop valence |
| 5 | `c0_minus_dist_val_col_pop` | 13 + 0p | -35374.0 | candidate | 1.6 ± 4.5 | 0.35 | drop candidate |
| 6 | `c0_minus_dist_val_col_pop_cand` | 12 + 0p | -35375.6 | avoidable | 19.4 ± 10.5 | 1.85 | drop avoidable |
| 7 | `c0_minus_avo_dist_val_col_pop_cand` | 11 + 0p | -35395.0 | load | 33.9 ± 10.1 | 3.38 | stop: the cheapest removal (load) costs 33.9 nats, 3.38 SE >= 2.0 |

**Lean model:** `c0_minus_avo_dist_val_col_pop_cand`, 11 population parameters + 0 person-level vectors. Kept: depth2, interior, confusion, singleton, features, typicality, attention, load, inhibit_leak, outlier. Removed: avoidable, distinct, valence, color, popout, candidate. Against the starting model (the full final): 20.8 ± 11.1 nats of ELPD-CV.

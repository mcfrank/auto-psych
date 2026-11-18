# ng (neighbourhood_doubt_three_example_memory): backward elimination

| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |
|---|---|---|---|---|---|---|---|
| 1 | `ng_full` | 27 + 0p | -12831.4 | seed | -1.8 ± 19.7 | -0.09 | drop seed |
| 2 | `ng_minus_seed` | 26 + 0p | -12829.5 | prime | -0.0 ± 2.0 | -0.02 | drop prime |
| 3 | `ng_minus_prime_seed` | 25 + 0p | -12829.5 | doubt_rich | 0.2 ± 3.3 | 0.07 | drop doubt_rich |
| 4 | `ng_minus_prime_seed_dr1` | 24 + 0p | -12829.7 | power | 0.7 ± 2.6 | 0.27 | drop power |
| 5 | `ng_minus_prime_pow_seed_dr1` | 23 + 0p | -12830.4 | doubt_band | 3.6 ± 2.4 | 1.51 | drop doubt_band |
| 6 | `ng_minus_prime_pow_seed_dr1_dband` | 22 + 0p | -12834.0 | range_high | 4.9 ± 3.8 | 1.28 | drop range_high |
| 7 | `ng_minus_prime_pow_hi_seed_dr1_dband` | 21 + 0p | -12838.9 | doubt_rich_multi | 6.2 ± 6.0 | 1.05 | drop doubt_rich_multi |
| 8 | `ng_minus_prime_pow_hi_seed_dr1_drm_dband` | 20 + 0p | -12845.1 | doubt_multi | -2.5 ± 1.3 | -1.93 | drop doubt_multi |
| 9 | `ng_minus_prime_pow_hi_seed_dr1_dm_drm_dband` | 19 + 0p | -12842.6 | square_cube | 11.2 ± 9.4 | 1.19 | drop square_cube |
| 10 | `ng_minus_prime_sq_pow_hi_seed_dr1_dm_drm_dband` | 18 + 0p | -12853.8 | parity_keep | 14.8 ± 11.0 | 1.35 | drop parity_keep |
| 11 | `ng_minus_prime_sq_pow_hi_seed_dr1_dm_drm_pk_dband` | 17 + 0p | -12868.6 | memory_limit | 20.9 ± 17.3 | 1.21 | drop memory_limit |
| 12 | `ng_minus_prime_sq_pow_hi_seed_dr1_dm_drm_pk_dband_mem` | 16 + 0p | -12889.5 | progression | 12.0 ± 9.5 | 1.26 | drop progression |
| 13 | `ng_minus_prime_sq_pow_hi_seed_prog_dr1_dm_drm_pk_dband_mem` | 15 + 0p | -12901.5 | contains_digit | 37.0 ± 26.3 | 1.41 | drop contains_digit |
| 14 | `ng_minus_prime_sq_pow_cd_hi_seed_prog_dr1_dm_drm_pk_dband_mem` | 14 + 0p | -12938.6 | retain | 54.1 ± 19.4 | 2.79 | stop: the cheapest removal (retain) costs 54.1 nats, 2.79 SE >= 2.0 |

**Lean model:** `ng_minus_prime_sq_pow_cd_hi_seed_prog_dr1_dm_drm_pk_dband_mem`, 14 population parameters + 0 person-level vectors. Kept: intercept, slope, parity, multiple, last_digit, round_multiple, range_low, span, retain, recency, primacy, doubt_single. Removed: prime, square_cube, power, contains_digit, range_high, seed, progression, doubt_rich, doubt_multi, doubt_rich_multi, parity_keep, doubt_band, memory_limit. Against the starting model (the full final): 107.2 ± 48.1 nats of ELPD-CV.

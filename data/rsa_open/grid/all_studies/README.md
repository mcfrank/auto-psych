# RSA grid on every study: PARTIAL results (2026-11-16)

Written by Claude Opus 5.5 from the Sherlock run of `docs/auto_open/RSA_GRID.md` (steps 3-5); not reviewed by a
person. Exploratory: the closed pre-registered data are spent.

**Partial.** These files are the report and page over the 82 of 97 fits that had finished at 16:16 PT on 11-16,
copied so the driver can look while the rest run. They will be replaced when every fit is done (or has timed out).

Still running when this was made (not in the table or the page):

- grid (6 of 66): `grid_d2_lstsingleton_graded_nocost`, `grid_d3_lstfc_graded_cost`, `grid_d3_lstfc_graded_nocost`,
  `grid_d3_shrsingleton_bool_nocost`, `grid_d3_shrsingleton_graded_cost`, `grid_d3_shrsingleton_graded_nocost`
- seed and discovered (9 of 31): `rsa_l1_shared_prior`, `isolated_graded_costly_l3`,
  `closed_c0e1_candidate_load_codability_listener`, `closed_c0e2_prototype_typicality_bounded_listener`,
  `closed_c0e3_boundary_avoidable_ambiguity_listener`, `closed_c1e3_feature_surprisal_ensemble_listener`,
  `closed_c2e1_bounded_endpoint_prev_redundancy_l2`, `cons_dup_sur_red_asp_int_comp_bind`,
  `open3_complexity_gated_retrieval_decision_l2`

The seed and discovered ones have a 48-hour limit (about 21:08 PT on 11-17); any that reach it will be reported as
not fittable in that time.

**Read "not converged" with care.** A diagnostic re-fit of three flagged theories (two grid, one discovered) found
every failing fold failing the same way: one of four chains stuck on the rating link of `pilot1_exp2`'s second
question (its center, spread and lapse: R-hat 1.53, bulk ESS 7), no divergences, and every theory parameter and every
other study's parameters converged (R-hat at most 1.004). That link belongs to `pilot1_exp2` alone, so the flag most
likely affects that column, not the others. The final report will check whether any ranking depends on it.

Files: `grid_report.{md,json}` (the table, scored against plain RSA `grid_d1_nosal_bool_nocost`, and the axis
contrasts), `grid_viz.html` (the page, 4.3 MB), one `<theory>.json` per fit (converged flag and seconds). The per-trial
`.npz` scores stay on Sherlock (`$SCRATCH/auto-psych/refits_1115/grid_fits/`).

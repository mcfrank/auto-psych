# RSA grid on every study: final results (2026-11-18)

Written by Claude Opus 5.5 from the Sherlock run of `docs/auto_open/RSA_GRID.md`; not reviewed by a person.
Exploratory: the closed pre-registered data are spent. The write-up is `docs/auto_open/RSA_GRID_REPORT.md`.

- **What is here:** the 36-theory grid (the cost axis was removed, PI 2026-11-17) and 27 of the 31 seed and discovered
  theories.
- **Not scored:** four closed incumbents timed out at 48 hours (`closed_c0e1`, `closed_c0e2`, `closed_c0e3`,
  `closed_c1e3`).
- **The 30 cost-theory fits** are left out of the report, the page and this folder.

Files:
- `grid_report.{md,json}`: each theory against plain RSA (`grid_d1_nosal_bool_nocost`), pooled and per study with
  clustered SEs, and the axis contrasts.
- `grid_viz.html`: the page, 3.4 MB.
- One `<theory>.json` per scored fit: converged flag and seconds.

**Convergence flags:** 24 of 36 grid fits and 11 of 27 others. In the three theories a diagnostic re-fitted, every
failing fold came from one chain stuck on `pilot1_exp2`'s second-rating link; the theories' own parameters converged.
The report gives every score with and without that study (§4 there).

The per-trial `.npz` scores stay on Sherlock (`$SCRATCH/auto-psych/refits_1115/grid_fits/`).

# Standard RSA, compared carefully on every reference-game study (2026-11-15)

Written by Claude Opus 5.5 at the PI's request; not reviewed by a person. A side question, not part of the discovery
loop: across the roughly 123,000 trials the campaigns hold (`DATA_INVENTORY.md`), which standard forms of RSA fit
best, and which of their choices (recursion depth, where salience enters, what it is, word meanings, utterance cost)
matter, study by study?

## The grid (`src/rsa_open/grid.py`, 66 theories)

Each grid point is a self-contained theory in the open campaign's format, so one fitting and cross-validation
pipeline scores it on every paradigm (choice, bets, ratings, production) through the campaign's links.

| axis | levels |
|---|---|
| depth | 0 (literal listener), 1, 2, 3 (pragmatic listeners, each inverting a speaker who simulates the level below) |
| where salience enters | none; the top listener's prior only (the speakers it simulates use a uniform one); shared by every level as common knowledge (the literal listener's prior too) |
| what salience is | feature count; being a singleton (no other object looks like it); one weight each |
| word meanings | Boolean; graded (a word is true of an object to 1 / its feature count) |
| utterance cost (depth ≥ 1) | none; a cost proportional to the share of the scene's objects a word is true of |

Depth 0 has no speaker (no cost) and one placement for salience: 6 theories. Depths 1-3: 2 × 5 × 2 = 20 each. Parameters:
`alpha` (rationality, depth ≥ 1), `w_salience`, `w_cost`; 0-3 per theory. A theory's speaker, for production and
judgment data, is the format's default (derived from its listener). On a prior trial (no word) a theory with salience
gives its prior. **The corners are the existing standard theories exactly** (tests): `literal_listener`, `rsa_l1`,
`rsa_l2`, `rsa_l1_salience` and `rsa_l1_shared_prior` (their familiarization weight at 0).

## The run

- **Here (cloud, committed data):** all 66 on the three committed literature sources (27,949 trials), grouped 5-fold
  CV, 4 chains × 500 + 500, as the stage-1 literature refit (`scripts/rsa/open_compute_checks/literature_refit.py`).
  Results in `data/rsa_open/grid/literature/` once done.
- **Sherlock (everything):** `scripts/rsa/open_compute_checks/rsa_grid.py`, below.

### Sherlock handoff

No agents, no Prolific; copies of the data, never the originals; aggregates only come back.

1. **The studies** (`studies.yaml`, see the script's docstring), from Sherlock's copies:
   - `literature`: the five sources' trial CSVs (the three committed, and Mayn & Demberg 2022 and 2023 from the
     ingest cache): one block E, temperature fixed at 1 as in the campaign;
   - `closed`: the nine chain experiments' `data/modelled.csv` and the pre-registered final's, labelled
     (`c0e1` … `c2e3`, `final`): one choice block;
   - `corpora`: open pilot 1's `experiment2/corpus.json` (prefix `pilot1`), stage 1's open chain
     `experiment3/corpus.json` (prefix `s1_open`), and its fixed chain's (prefix `s1_fixed`, `merge: true`: one
     choice block for the three experiments).
   ```bash
   python scripts/rsa/open_compute_checks/rsa_grid.py corpus --studies studies.yaml --out corpus.json
   python -m src.rsa_open.grid grid
   ```
   Report the corpus's blocks, trials, units and studies (the script prints them), against `DATA_INVENTORY.md`.
2. **Fits:** 66, one per grid point, at the loop's settings (1,000 + 1,000, 5 folds per block):
   `rsa_grid.py fit --corpus corpus.json --grid grid --index $i --out fits`. An array of 66 single-core tasks (16 GB
   each; depth 3 is the slowest), or a 24-core job with `xargs -P 20`.
3. **Report:** `rsa_grid.py report --corpus corpus.json --grid grid --out fits` writes `grid_report.{md,json}`: each
   theory against plain RSA, pooled and per study with clustered SEs, and each axis's contrast (pairs of grid points
   that differ only on it), per study. Copy both to `data/rsa_open/grid/all_studies/` and write
   `docs/auto_open/RSA_GRID_REPORT.md`: the table, the axis contrasts, any theory that did not converge, and time.

**Caveats for reading it.** The closed campaign's pre-registered data are spent for confirmatory claims
(`docs/auto_rsa/CONSOLIDATION.md` §5): this is exploratory. Studies differ in paradigm and in trials per person, so the
per-study columns matter as much as the pooled one. Stage 1's two chains share people.

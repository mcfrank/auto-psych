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
4. **Page:** `python scripts/rsa/open_compute_checks/grid_viz.py --corpus corpus.json --grid grid --fits fits --out
   grid_viz.html`, copied to `data/rsa_open/grid/all_studies/` (aggregates only; check its size, it grows with the
   displays: about 2 KB a display).

**Caveats for reading it.** The closed campaign's pre-registered data are spent for confirmatory claims
(`docs/auto_rsa/CONSOLIDATION.md` §5): this is exploratory. Studies differ in paradigm and in trials per person, so the
per-study columns matter as much as the pooled one. Stage 1's two chains share people.

## First results: the committed literature (all 66)

Three sources, 27,949 trials, 164 units; grouped 5-fold CV, 4 chains × 500 + 500. Nats over plain RSA
(`grid_d1_nosal_bool_nocost`), clustered SE ratio in parentheses; the full table is
`data/rsa_open/grid/literature/grid_report.md`.

| theory | pooled | Mayn & Demberg 2026 | pragmods | Sikos 2021 |
|---|---|---|---|---|
| d2, shared singleton salience, Boolean | +445 | +9 (0.7) | +15 (0.5) | +420 (3.8) |
| d1, shared singleton salience, graded | +443 | +7 (0.5) | +16 (0.6) | +420 (3.8) |
| d2, shared singleton, graded, cost | +441 | +10 (0.7) | +11 (0.4) | +419 (3.8) |
| d3, top-listener singleton, graded, cost (best at depth 3) | +440 | +13 (0.8) | +10 (0.4) | +417 (3.9) |
| ... the 19 converged depth-1 to 3 singleton theories lie within +396 to +445 | | | | |
| d0, listener singleton salience, graded | +186 | −125 (−2.9) | −111 (−3.1) | +422 (3.9) |
| best without salience: d2, graded, cost | +116 | +9 (0.8) | +28 (1.6) | +79 (2.6) |
| d2, no salience, Boolean (plain RSA, depth 2) | +39 | +7 (1.5) | +14 (2.3) | +17 (3.2) |
| literal listener | −864 | −429 (−3.0) | −512 (−4.5) | +77 (3.0) |
| *for comparison: the 6-parameter seed (stage-1 refit)* | *+594* | *−7* | *+171* | *+430* |

Each axis, averaged over pairs of grid points that differ only on it (wins: pairs where the second level is better;
depth-0 pairs left out except for depth itself, and pairs with a non-converged fit left out, as the page does by default):

| contrast | pairs | pooled | wins | Mayn & Demberg 2026 | pragmods | Sikos 2021 |
|---|---|---|---|---|---|---|
| depth 1 − 0 | 6 | +473 | 6/6 | +230 | +295 | −52 |
| depth 2 − 1 (converged pairs) | 17 | +26 | 17/17 | +5 | +8 | +13 |
| depth 3 − 2 (converged pairs) | 16 | +7 | 13/16 | +5 | −2 | +4 |
| singleton salience − none (depth ≥ 1, top listener / shared) | 8 / 8 | +370 / +368 | 8/8, 8/8 | +6 / −3 | −6 / −4 | +371 / +376 |
| feature-count salience − none (depth ≥ 1, top listener / shared) | 8 / 8 | −21 / −19 | 0/8, 0/8 | +4 / 0 | −20 / −18 | −5 / 0 |
| shared − top-listener salience (same kind) | 8 + 8 | +2 (fc), −2 (singleton) | 3/8, 7/8 | | | |
| graded − Boolean meanings, depth 1 / depth 2 | 10 / 10 | +57 / +2 | 10/10, 9/10 | +3 / −2 | +25 / +2 | +29 / +2 |
| (graded − Boolean at depth 0, the literal listener) | 3 | +622 | 3/3 | +216 | +401 | +5 |
| extension cost − none | 20 | +34 | 15/20 | +1 | +5 | +27 |

What it says so far:

1. **One choice dominates: salience as being a singleton**, worth about 370 nats, and all of it on Sikos 2021, whose
   displays turn on a unique object. On Sikos a literal listener with that prior (+422) does as well as any pragmatic
   one, and plain RSA is *worse* there than the literal listener (depth 1 − 0: −52). Feature-count salience never helps
   once there is a speaker (0 of 16 pairs).
2. **Pragmatic reasoning matters on the other two sources** (depth 1 over 0: +230 and +295), but **a second level adds
   little** (+26, though in 17 of 17 converged pairs) **and a third almost nothing** (+7, 13 of 16). Whether salience is common knowledge or the top listener's alone makes no
   difference here (±2).
3. **Graded meanings help a depth-1 listener** (+57, 10 of 10 pairs, on pragmods and Sikos) **and do almost nothing
   at depth 2** (+2): a second level of reasoning and graded meanings do much the same work here, so the open arm's
   recurring discovery is, on this literature, an alternative to deeper recursion rather than an addition to it. (An
   earlier version of this section said +107 on every source: that average included the literal listener, for which
   graded meanings are a large repair, +622.) Extension cost helps a little (+34, 15 of 20).
4. **On Mayn & Demberg 2026, the largest source, no choice beyond depth 1 matters** (every depth-1/2 theory within
   ±16 nats of plain RSA).

**The page.** `data/rsa_open/grid/literature/grid_viz.html` (built by `scripts/rsa/open_compute_checks/grid_viz.py`
from the fits; aggregates only) shows the grid as a heatmap per source, these contrasts computed live, and all 213
displays with people's choices against any two grid theories, sorted by where they differ. It is the campaign's
old model-report design (`src/rsa/report_template.html`) carried over. For the Sherlock run, build it from the
all-studies fits the same way; display panels cover the choice blocks only.
5. **The best grid theory is about 150 nats behind the 6-parameter seed, and the gap is pragmods** (+15 against +171):
   pragmods' prior and salience conditions carry preferences the grid's one-weight salience cannot express. On Sikos
   and Mayn & Demberg the grid matches the seed.

Not converged (R-hat or divergences; read with care), 7 of 66: `grid_d1_lstfc_bool_cost`,
`grid_d1_lstsingleton_bool_cost`, `grid_d2_shrsingleton_graded_nocost` and **all four depth-3 theories with a shared
singleton prior**. Their Mayn & Demberg scores (−68 at depth 2; −95 to −412 at depth 3) are far out of line with their
converged neighbours (the top-listener versions score +10 to +14), so they are fitting failures, not evidence against
the theories: a prior repeated through four levels makes the salience weight and alpha hard to separate. Counting them,
depth 3 − 2 reads −46; without them, +7. On Sherlock, at 1,000 + 1,000, check whether they converge before reading
that cell. Fits took 4-28 minutes each on this container.

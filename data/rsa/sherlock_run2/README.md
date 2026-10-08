# RSA inner loop, Sherlock run 2 (7–8 October 2026)

The run reported for claim 1 ("runs on existing data yield candidates better
than the starting models on held-out conditions within the same papers").
Results and narrative: `docs/auto_rsa/SHERLOCK_RUN2_RESULTS.md`; the report
across runs 1 and 2: `data/rsa/existing_data_report.html`; the sweep overview:
`overview.html` (built by `python -m src.rsa.sweep_report --sweep data/rsa/sherlock_run2`).

## Design (fixed before the run: `docs/auto_rsa/PLAN.md`, "Decisions 2026-10-08, after run 1")

- Code: `dec3790ebf8e0c1ce99db8536d1b21e5e1a2818c` (each cell's `code_commit`).
- Data: five sources, forced-choice listener trials; conditions held out within each paper
  (`src/rsa/split.py`). Hashes of the cells' `train.csv`, `test.csv`, `split.json` are in each
  `data.sha256`.
- Seven cells: `real_rep1-3` (3 replicates × 8 rounds), `recovery_literal_rep1-2` and
  `recovery_salience_rep1-2` (2 × 4 rounds; choices simulated from the withheld ground truth).
  Six Gemini 3.8 Flash agents per round, no network access; selection, pruning and export on
  grouped 5-fold CV over training conditions; live-set cap 12. Each cell's settings: `cell.json`.

## Files per cell (`<cell>/`)

| file | what it holds |
|---|---|
| `cell.json`, `code_commit`, `data.sha256` | the cell's settings, code and data hashes |
| `history.json` | every scoring step: the best model, every live model's standing (PSIS-LOO, grouped CV overall and per source), the round's events, the live set and ledger length |
| `attempted_hypotheses.jsonl` | the ledger: every candidate (admitted / rejected with the reason) and every prune (with the margin), each with its full hypothesis |
| `models/`, `models/pruned/` | every admitted model's source; `models_manifest.yaml` lists the live set at the end |
| `best_model.py`, `export.json` | the exported model and the final live set |
| `round_<k>/candidate_<i>[_retry_1\|_repair_1]/` | each agent's `candidate.py`, `hypothesis.md`, `model_name.txt` (agent logs are archived, not here) |
| `round_<k>_abandoned_<n>/` | a round interrupted by the midnight crash and rerun (`SHERLOCK_RUN2_RESULTS.md` §2) |
| `heldout/heldout.csv`, `heldout.json` | claim 1: every model's held-out lpd, overall and per source, vs the best starting model, SE clustered by held-out condition |
| `heldout/report.html`, `report.bundle.json` | people against the models on each held-out display (choice counts per display only) |
| `heldout/unit_lpd.csv` | the shown models' lpd summed per unit: held out (`test`), in sample (`train`), out of fold (`cv`) |
| `recovery/` (recovery cells) | `recovery.json`: the verdict, the exported, closest-live and best-seed distances to the ground truth |
| `.cv/folds.json` | the CV folds' record (the fold files are trial-level and archived) |
| `novelty_pool.json` | the displays the novelty gate and the recovery distances use |
| `report.html`, `report.bundle.json` | the loop's own report (training data, by display) |
| `token_usage.jsonl`, `token_usage_summary.json` | every agent call's tokens and cost; the summary covers the whole log (regenerated 2026-10-08 for the three resumed cells, whose summary had covered only the last process) |
| `agent_activity.md` | web tool calls (none in run 2) and the URLs and outside paths that appear in the agents' logs |

Only aggregates are committed: no trial rows, no participant identifiers, no
fit caches, no agent logs.

## Archived on Sherlock

`scripts/rsa/slurm/archive_sweep.sbatch` copies what the repository does not hold off
`$SCRATCH` (purged after 90 days): the prepared data and recovery simulations (trial-level,
private), the cell records and Slurm logs, each loop directory with the agents' logs and
opencode sessions, and the fit caches. It records `SHA256SUMS` and `MANIFEST.json`; copy both
into `archive/` here once it has run, with the archive's location.

Archived 2026-10-08 (job 47042589) to `/oak/stanford/groups/mcfrank/auto-psych/archive/rsa_run2` ($OAK), 5.1 GB; checksums and manifest in `archive/`.

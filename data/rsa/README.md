# RSA runs: records brought back from Sherlock and the cloud

Every directory here is a record of a run in the RSA campaign (`docs/auto_rsa/CAMPAIGN_LOG.md`
§4 lists them in order, with what each found). They hold aggregates, designs, model files,
loop histories, reports and agent transcripts. They never hold Prolific participant ids or
trial-level live data: those stayed on Sherlock, in each cell's `private/` directory. A
`private/` directory *here* holds only scores (`prospective.json`, `recovery.json`) and the
run's `outer_config.json`.

Trial-level literature data are not here either. The CC-BY sources and pragmods are under
`src/pipelines/outer_loop/projects/rsa_reference/data/`. Sources without a licence are fetched
into `external/` (gitignored) by `src.rsa.ingest`; `combined_trials.csv` is rebuilt by
`src.rsa.ingest.combine` and also gitignored.

| path | what | doc |
|---|---|---|
| `seed_comparison_all/`, `report.html`, `report.bundle.json` | the five starting models compared on pragmods (2026-10-06) | `PLAN.md` |
| `loop_smoke/` | smoke test 1: one round of the inner loop on pragmods | `SMOKE_RESULTS.md` |
| `loop_smoke2/`, `loop_smoke2_seeds/`, `loop_smoke2_repair/` | smoke test 2: all sources, sandboxed agents, a repair attempt | `SMOKE_RESULTS_2.md` |
| `sherlock_run1/` | the first Sherlock sweep (real data and recovery cells) | `SHERLOCK_RUN1_RESULTS.md` |
| `sherlock_run2/` | claim 1: held-out tests on published data, and recovery (its own `README.md`) | `SHERLOCK_RUN2_RESULTS.md` |
| `existing_data_report.html` | the existing-data phase report (runs 1 and 2) | |
| `live_seeds/` | the promoted seeds (`models/`), their selection (`promotion.json`), the chains' split (`chains/`) and an early design (`design_t10/`) | `HANDOFF_sherlock_promote.md` |
| `rehearsal/`, `rehearsal2/`, `rehearsal3/` | dress rehearsals of the live loop on simulated participants | `REHEARSAL_REVIEW.md`, `REHEARSAL2_REPORT.md`, `REHEARSAL3_REPORT.md` |
| `live/live_rsa_c{0,1,2}/` | the campaign: three chains × three live experiments (designs, loop histories, exported models, claim-2 scores in `private/`) | `CAMPAIGN_C*_REPORT.md` |
| `live/finals/` | the three chains' final models | `PREREG_FINAL.md` |
| `live/finals_pool_preds_final.json`, `live/bar_pool_preds.json` | **frozen** predictions of the finals and of the 17 bar models on the 1,597-display pool (hashes on OSF s48jv; never edit) | `PREREG_FINAL.md` §8, `OSF_PREREG.md` |
| `live/finals_pool_preds.json`, `live/finals_pool_fit.{py,sbatch}` | an earlier pooled fit of each chain's latest best, superseded by the frozen file above; the script is kept as its provenance | `CAMPAIGN_LOG.md` |
| `prereg/` | the pre-registered experiment: power (`power.json`), the **frozen** design (`design/`), its power (`design_power.json`) and the result (`run/prereg_analysis.json`) | `PREREG_FINAL.md`, `PREREG_RESULT_REPORT.md` |
| `consolidate/` | the consolidation: the elimination (`eliminate/`, with the lean model file) and the prospective score (`score.json`) | `CONSOLIDATION.md` |
| `viz/panels.json` | per-display choice counts and pooled-fit predictions, for the figures | `docs/auto_rsa/figures/README.md` |

Docs are in `docs/auto_rsa/` unless a path is given.

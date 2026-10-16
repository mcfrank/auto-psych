# Campaign figures

Made by `uv run python -m src.rsa.viz.figures` from summaries in the repository; no fitting.
Colors (published / campaign / pre-registered, and the categorical slots elsewhere) are the
reference palette of the dataviz guidance.

| figure | what it shows | data |
|---|---|---|
| `fig1_existing_data.png` | A: run 2's held-out published data, every model vs the best starting model, exported model with 95% CI. B: recovery, each model's pool RMSE to the simulated ground truth (4 of 4 recovered) | `data/rsa/sherlock_run2/` |
| `fig2_chain_trajectories.png` | per chain and live experiment, the incumbent's selection ELPD (grouped CV on the data so far) by loop round, relative to the experiment's starting set; rings mark a new incumbent | `data/rsa/live/live_rsa_c*/outputs/experiment*/model_loop/history.json` |
| `fig3_prospective_tests.png` | A: claim 2, each committed model on the next experiment vs the best starting model and the best promoted seed. B: the pre-registered test (OSF s48jv), each final vs the best promoted seed, overall and by design half | `prospective.json`, `data/rsa/prereg/run/prereg_analysis.json` |
| `fig4_prereg_all_models.png` | every model's log predictive density on the pre-registered data, relative to the best promoted seed | `prereg_analysis.json` |
| `fig5_model_vs_data.png` | model vs data per display and object class: chain 0's final, the best promoted seed and the best starting model, colored by dataset | `data/rsa/viz/panels.json` (see below) |
| `fig6_consolidation.png` | A: the backward elimination, each mechanism's removal cost in SEs by round. B: the consolidated models scored on the pre-registered data vs chain 0's final | `data/rsa/consolidate/` |

**Figure 5's data.** The pre-registered experiment's per-display counts exist only on Sherlock.
`sbatch scripts/rsa/slurm/viz_export.sbatch` (`src.rsa.viz.export_panels`) writes
`$SCRATCH/auto-psych/rsa_consolidate/panels.json`: each display's choice counts per class, for the
published, campaign and pre-registered data, and the posterior-mean class probabilities of each
model fitted on the pooled data (the fit behind the frozen predictions). It holds aggregates
only: no trial rows, no participant ids. Bring it back as `data/rsa/viz/panels.json` and rerun the
figures.

Until then, figure 5 is an **interim** version (it says so in its subtitle): observed counts from
the chains' report bundles and the frozen pool predictions, for the displays in the 1,597-display
pool only (the published studies' plain displays and all campaign displays), without the
pre-registered data.

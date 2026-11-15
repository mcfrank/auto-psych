# Handoff: every stage-1 theory on every stage-1 paradigm (2026-11-15)

For the Sherlock session. Written by Claude Opus 5.5; not reviewed by a person. No agents, no Prolific; aggregates
only come back.

**Why.** Stage 1 is now a pilot (the gate failed on open experiment 2's information item, `STAGE1_REPORT.md`), and
the PI wants to understand what was learned, not only what the protocol says. Each chain scored its theories only on
its own blocks. This refits the theories that matter on **all seven blocks at once**: the literature E, the open
chain's choice, dual rating and production, and the fixed chain's three choice experiments. It is a dry run of the
bakeoff's fit A (`PREREG_V2.md` §6) on pilot data, descriptive only. The questions:

- Do the fixed arm's theories, built on one choice paradigm, hold up on the open arm's rating and production data?
- Do the open arm's hold up on the fixed paradigm's choices?
- How much better than plain RSA and the seeds is each, block by block?

What the driver found locally, to compare against (`STAGE1_ANALYSIS.md`): the fixed arm's theories grew into case
tables over display types (feature-tier indicators in 80% of them, median 136 lines, the final 205 lines with one
parameter), the open arm's stayed graded-semantics RSA with speaker models (median 71 lines).

## The run

On `auto-psych-open` at this handoff's commit or later, from a clean checkout in a scratch work dir:

1. **Corpora:** each chain's experiment 3 corpus, copied, never modified:
   `$SCRATCH/auto-psych/rsa_open_v2/.../_runs/chain_0/experiment3/corpus.json` for `open_chain_0` and
   `fixed_chain_0` (the run trees under `agent_trees/<id>/repo/_runs/chain_0`). The script checks that both hold the
   same E.
2. **Theories** (`theories.tsv`: `<name>\t<path>`; 13 lines). The paths are the run trees' own model files
   (`experiment<n>/loop/models/` or its `pruned/`):
   - `chance` (write `src.rsa_open.loop.orchestrator.CHANCE_THEORY` to a file) and the references from
     `data/rsa_open/seeds/bar/`: `literal_listener`, `rsa_l1`, `rsa_l2`, `rsa_l2_singleton_feat_color_valence_l0`,
     `competitor_confusion_speaker`, `softmax_surprisal_isolated_costly_l4`;
   - each chain's incumbent after each experiment: open `extension_noisy_graded_semantics_l2` (exp 1),
     `l1_graded_semantics_l2` (exp 2), `complexity_gated_retrieval_decision_l2` (exp 3); fixed
     `prototype_tier_confusion_l2`, `contrastive_anchor_decision_l2`, `sole_exclusion_shadowing_heuristic`.
   Prefix the incumbents with their chain and experiment (`open3_complexity_gated_retrieval_decision_l2`) so the table
   reads by itself.
3. **Fits:** one per theory, at the loop's settings (4 chains × 1,000 + 1,000, 5 folds per block, the defaults):
   ```bash
   python scripts/rsa/open_compute_checks/cross_refit.py fit --open-corpus open.json --fixed-corpus fixed.json \
     --theories theories.tsv --index $i --out cross_out
   ```
   A single job on `mcfrank` (24 CPUs, 96 GB, 12 h) running the 13 with `xargs -P 12`, or a 13-task array of 1 CPU
   and 8 GB each: each theory is 6 fits (full and 5 folds) on about 80k trials; expect 1-3 h per theory.
4. **Table:** `cross_refit.py report ...` writes `cross_out/cross_refit.{json,md}`.

## Report

In a new file `docs/auto_open/CROSS_REFIT_REPORT.md`, with `cross_refit.md` and `cross_refit.json` copied to
`data/rsa_open/live_v2/cross_refit/` (aggregates: per-theory, per-block elpd, gains over chance and over RSA with
clustered SEs, and per-display gain summaries; not the `.npz` per-trial scores, which stay on Sherlock):

1. The table as the script prints it, and any theory that did not converge (with its R-hat and divergences).
2. For each block: the best theory, and whether the other arm's final is within 2 SEs of it.
3. Time and memory per theory.

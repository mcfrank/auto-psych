# Handoff: rehearsal 2 (one guarded run, two experiments, a coherent ground truth)

From the driver session, 2026-10-10, for the local Sherlock session.
Context: `REHEARSAL_REVIEW.md` and the PI's decisions of 2026-10-10 (`PLAN.md`).

**Launch signal:** a commit on `auto-rsa` whose message starts `RSA: rehearsal 2 ready`.

## 0. Pull

```bash
cd ~/auto-psych && git fetch origin && git merge --ff-only origin/auto-rsa
```

`git fetch origin auto-rsa` leaves `origin/auto-rsa` stale on git 1.8: fetch the remote, then merge.

## 1. What changed since rehearsal 1

- **Fit limits.** Only a new candidate's admission fit is limited to 30 minutes. Seeds,
  carried and admitted models, and the outer loop's bar models and ground truth fit
  without a limit, still one per pinned child (`fitting.NO_TIME_LIMIT`). This fixes
  rehearsal 1's crash.
- **The guarded selection rule** (`OuterConfig.selection_scope = "guarded"`, the
  default). Every fit uses all data. A model is eligible only within 4 clustered SEs of
  the best on the existing data's grouped CV, and eligible models are ranked on the
  live trials. Ineligible models are pruned at the end of each inner loop, with a ledger
  line `ineligible: ...`.
- **The design also aims at the bar.** Experiment N's design is over the carried models
  plus the five starting models and the 12 promoted seeds (`bar:<name>`):
  - half the prior is on the carried models;
  - a bar model that is the same file as, or the same hypothesis as, a model already in
    is merged;
  - the ground truth is withheld from the bar, unnamed (`eig.json` `n_withheld`).
- **Carry-forward keeps one model per hypothesis the live displays can tell apart**
  (`experiment<N>/carry.json`, at most 8).
  - "Same" means within 0.002 RMSE on the plain-display design pool.
  - That threshold was calibrated on human-data fits: rehearsal 1's `twin_*` copies were
    a median 0.0002 apart, while `rsa_l1` vs `rsa_l2` are 0.0070.
- **Recovery** is now the mean KL divergence from the ground truth (`kl_pool`, and
  `kl_design` on that experiment's designed displays). Pool RMSE is kept beside it.
- **Agent spend:** each `model_loop/` has `token_usage.jsonl` and
  `token_usage_summary.json`, and the run has `token_usage_summary.json` (per experiment
  and total).
- **Job headers have no `--error`:** stderr follows `-o`.
- **Plain-display scope** (PI 2026-10-10): the run fits the existing data without
  pragmods E5/E6/E7 and the colour-prior rerun (`<run>/existing_scope.json` lists what
  was left out). Novelty is measured on plain displays, there is no framing lens, and the
  agents' context says so.
- **Claim 2 is the committed model's test:** `prospective.json` `committed_vs`, from
  experiment 2 on.
- **The ledger carries across experiments.**

## 2. Submit (after a pull; ~15 h)

```bash
mkdir -p "$SCRATCH/auto-psych/rsa_rehearsal2/logs"
sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_rehearsal2/logs/%x_%A_%a.out" \
  scripts/rsa/slurm/outer_rehearsal.sbatch
```

- **Defaults:**
  - one array task, `SCOPES=guarded`;
  - `N_EXPERIMENTS=2`, `PARTICIPANTS=200`, `DISPLAYS=40`;
  - chain 0's seeds;
  - its own `WORK_ROOT`, `$SCRATCH/auto-psych/rsa_rehearsal2`, so rehearsal 1's tree is
    untouched.
- **The ground truth** (`GT_RULE=coherent`, `src.rsa.outer.ground_truth`):
  - among run 2's admitted models that were not promoted, converged and within 150 nats
    of the best on grouped CV over all existing data (`promotion.json`);
  - pick the one farthest (design-pool RMSE) from its nearest chain-0 seed;
  - written to `$WORK_ROOT/ground_truth.json`.

  Its file is in `data/rsa/sherlock_run2/<cell>/models/`, outside every agent tree.
  `check_agent_tree.sh --gt-file` refuses any copy of it, or any file named after it, in
  the tree.
- **Fits:** promote's cache serves the ground-truth search (default sampler settings). The
  run itself fits with dense mass, so its first design refits the chain's seeds and the
  bar models once (minutes each).

## 3. While it runs

- **Thread check** at about 10 minutes:
  `srun --jobid=<id> --overlap -n1 ps -u $USER -o pid,nlwp,pcpu,psr,cmd --sort=-nlwp | head -30`.
- **After experiment 1's inner loop:** `experiment2/carry.json` should list at most 8
  distinct models, and `experiment2/design/eig.json` should show more than about 3 bits of
  joint EIG and power well above rehearsal 1's 0.25.

## 4. Bring back (into `data/rsa/rehearsal2/<cell>/`, as for rehearsal 1)

- `outputs/` without trial-level CSVs (`experiment*/data/*.csv`, `model_loop/responses.csv`,
  `.cv/fold_*.csv`).
- `private/` without the fit cache.
- `ground_truth.json` once, at `data/rsa/rehearsal2/`.
- The job log.

## 5. Report per experiment

- **The design:** the models in it (carried / bar / merged), joint EIG, and power.
- **Claim 2 (experiment 2):** `committed_vs`, the committed model against the best starting model, the best
  promoted seed and each bar model; and `live_vs` as the secondary measure.
- **Recovery:** the exported model's `kl_pool` and `kl_design`, and the closest model
  before and after.
- **`carry.json`:** kept, merged and over the cap.
- **Ineligible prunes:** ledger lines starting `ineligible:`.
- **Wall time and agent spend:** `token_usage_summary.json`.
- **Any failure.**

# Handoff: the pre-registered final experiment (for the Sherlock session)

From the driver session, 2026-10-15. The plan is `PREREG_FINAL.md`; the decisions are
`CAMPAIGN_LOG.md` #58–#61.

There is **one study** of new people. Each final model is tested against the best promoted seed,
on frozen predictions. There are no agents and no critic.

**Where it runs:**
- **A new work root:** `$SCRATCH/auto-psych/rsa_prereg`, with its own staged code and
  `code_commit`. The campaign's root and its frozen code are not touched.
- **Copied from the campaign, once:** the pooled data and the finals' fits, from
  `rsa_campaign/finals_pool/`.

**Rules, as in the campaign:**
- Never print keys or tokens.
- Prolific ids stay in `cells/prereg/private/` and never leave Sherlock.
- Bring back no trial-level data, only aggregates and records.
- Live recruitment needs the PI's go-ahead, the config flag, `CONFIRM_LIVE_RECRUITMENT=1` and the
  typed "yes".

## Steps

**1. Pull `auto-rsa`** into `$HOME/auto-psych` (no local changes).

**2. Fit the 17 bar models** on the finals' pooled data: about 1–3 h on 8 cores. The two copies
of `rsa_l1` and `rsa_l2` share one cache entry.

```bash
cd $HOME/auto-psych
STAGE=fit_bar sbatch scripts/rsa/slurm/prereg_prepare.sbatch
```

- This copies `rsa_campaign/finals_pool/{pooled.csv,.fit_cache}` into `rsa_prereg/pool/`, then
  writes `rsa_prereg/bar_pool_preds.json`.
- **Check:** all 17 converged. Look for `"converged": true` in each model's `fit`, and the
  "done …" lines in the log.
- **Bring back:**
  - push the file to `auto-rsa` as `data/rsa/live/bar_pool_preds.json` (model predictions on
    the pool: no trial data);
  - report the convergence, and anything that needed the refit.

**3. Power: the driver runs it** in the cloud from the two predictions files (fast, numpy only).
- If you would rather run it here: `STAGE=power sbatch scripts/rsa/slurm/prereg_prepare.sbatch`
  writes `rsa_prereg/power.json` (grid G ∈ {104, 208, 312, 416} × N ∈ {800, 1200, 1600, 2400}).
- The PI then picks G and N (`PREREG_FINAL.md` §7).

**4. Make the design, once,** with the chosen numbers. It runs the campaign's design rule over 20
models (every fit cached), then the random half, then the design's own power.

```bash
STAGE=design DISPLAYS=<G> PARTICIPANTS=<N> sbatch --time=36:00:00 scripts/rsa/slurm/prereg_prepare.sbatch
```

- **Expected time:** EIG selection of G/2 picks takes about 2–6 h at G = 312.
- **It refuses to run twice.** A second design would be a forking path. If it must be redone,
  say why in the campaign log first.
- **Bring back** to `data/rsa/prereg/design/` and push. None of these contain participant data:
  - `design/design.json`, `design/trial_lists.json`, `design/prereg_design.json`;
  - `design/eig_half/eig.json` and `design/eig_half/design_d<G/2>.json`;
  - `design_power.json`.

**5. The freeze is the driver's step,** then the PI's:
- the driver fills `PREREG_FINAL.md` §8 and `rsa_prereg.yaml` (`participants`, `displays`,
  `design_sha256`, both predictions' `sha256`), and pushes the freeze commit;
- the PI approves it.
- **Pull it.**

**6. Test deploy:** the page and a Prolific draft. Nothing is published. This stages the frozen
code as `rsa_prereg/harness_repo` (with `code_commit`); the prepare stages used their own copies
(`rsa_prereg/code_<stage>/`). A commit after this one (docs only, say) needs
`RESTAGE_REASON` at the live launch, as in the campaign.

```bash
PROLIFIC_MODE=test bash scripts/rsa/live/prereg_launch.sh
```

- **Check:** the page at `auto-psych-2c5da-prereg-c0.web.app/…`. A new Hosting site: check that
  Safe Browsing does not flag it (`SAFE_BROWSING_C2.md`).
- **Check the draft:** N places, $0.50, 3 min, and the blocklist covering every earlier RSA
  study.
- **Check the balance:** the Prolific account covers N × $0.50 × 1.33. Chain 0's experiment 3
  failed to publish on the balance.

**7. Live,** only on the PI's go-ahead:
- set `confirm_live_recruitment: true` in a copy of `scripts/rsa/live/rsa_prereg.yaml` outside
  the checkout (an edited tracked file makes the checkout dirty, and staging refuses it);
- run `PROLIFIC_MODE=live CONFIG=<the copy> bash scripts/rsa/live/prereg_launch.sh` and type
  `yes`.
- The job polls for up to 48 h (`--time=72:00:00`), and resumes on a resubmit without
  republishing.
- About 1,200 people may take most of a day. Someone must clear Prolific's held submissions as
  they come.

**8. Bring back:**
- `cells/prereg/run/experiment1/prereg_analysis.json`;
- `cells/prereg/run/experiment1/data/participants.json` (counts and the excluded ids are run ids,
  not Prolific ids: check);
- the job log's summary lines.
- **Not** the CSVs, `raw_collected/`, `participant_ids.json` or anything in `private/`.
- Scan for keys and 24-character Prolific ids before pushing, as before.

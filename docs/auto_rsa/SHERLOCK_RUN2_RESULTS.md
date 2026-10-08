# RSA inner loop, Sherlock run 2: results

Run 2026-10-07/08 by the local session that followed `HANDOFF_sherlock_run2.md`.
Per-cell outputs are in `data/rsa/sherlock_run2/<cell>/`: run 1's §6 file set, plus
`.cv/folds.json` and the §7 held-out pages (`heldout/report.html`,
`report.bundle.json`, `unit_lpd.csv`). The overview page is
`data/rsa/sherlock_run2/overview.html`. Trial-level data (including the CV folds'
`fold_*_train.csv`), fit caches and agent logs stay on Sherlock in
`$SCRATCH/auto-psych/rsa_run2/`.

## Claim 1: are the exported models better than the starting models on held-out conditions?

**Yes, in all three real cells, by about 2-2.4 clustered SE each.** All three
gains come from two of the five sources.

| cell | exported model | held-out lpd vs best seed (rsa_l2) | held-out rank | by source (exported - rsa_l2) |
|---|---|---|---|---|
| real_rep1 | rsa_l2_singleton_feat_color_valence_l0 | **+84.3** (SE 35.7) | 2/58 | pragmods +66.4, sikos_2021 +16.2, Mayn & Demberg 2022/2023/2026 -0.2/+2.2/-0.3 |
| real_rep2 | surprisal_isolated_graded_costly_l5 | **+56.2** (SE 24.1) | 8/57 | pragmods +34.7, sikos_2021 +21.7, M&D -0.5/+0.7/-0.3 |
| real_rep3 | crowd_discrim_confusion_chromatic_l2 | **+26.4** (SE 13.1) | 11/58 | pragmods +9.9, sikos_2021 +15.4, M&D -0.4/+1.9/-0.3 |

- Held-out data: 10,632 trials in 84 held-out units (training conditions' papers,
  other conditions).
- Every model the cell ever admitted is scored there, pruned ones included. "Rank" is
  among all of them.
- On the three Mayn & Demberg datasets, no exported model moves held-out lpd by more
  than 2.2 from rsa_l2.

## How the models improve over rounds

Each row is the step's best model by ELPD-CV (5 folds of whole training conditions),
with its held-out lpd minus rsa_l2's. Round 0 is the first scored step after the seeds.

| round | real_rep1 CV / held-out | real_rep2 CV / held-out | real_rep3 CV / held-out |
|---|---|---|---|
| seeds | -17576.6 / 0 | -17577.9 / 0 | -17561.0 / 0 |
| 0 | -17576.6 / 0 (rsa_l2) | -17483.6 / +18.7 | -17199.2 / +18.1 |
| 1 | -17187.5 / +21.3 | -17455.4 / +16.6 | -17170.4 / +21.2 |
| 2 | -17159.6 / +28.6 | -17360.0 / +28.3 | -17170.4 / +21.2 |
| 3 | -17157.1 / +26.4 | -17355.7 / +26.7 | -17166.9 / +22.7 |
| 4 | -17150.3 / +39.1 | -17355.7 / +26.7 | -17166.1 / +27.3 |
| 5 | -17146.7 / +44.7 | -17325.5 / +59.1 | -17166.1 / +27.3 |
| 6 | -17146.7 / +44.7 | -17278.3 / +58.2 | -17141.3 / +20.3 |
| 7 | -17129.1 / +84.3 | -17271.1 / +56.2 | -17131.6 / +26.4 |
| 8 (end) | -17129.1 / +84.3 | -17271.1 / +56.2 | -17131.6 / +26.4 |

- **The real cells keep improving to round 7, and none improves at round 8.** The best
  held-out gain arrives late in real_rep1 (round 7) and real_rep2 (round 5).
- **Each replicate refines one lineage of mechanisms rather than jumping between
  families:**
  - real_rep1: rsa_l2 + singleton prior → distinctiveness → features → colour →
    valence;
  - real_rep2: distinctive object → isolated/graded → costly → surprisal;
  - real_rep3: singleton isolation → crowding → chromatic → confusion → discrimination.
- **CV and held-out mostly move together, but not always.** real_rep2 rounds 5→7 and
  real_rep3 round 6 improved CV while held-out fell by 1-7.
- **Across replicates, CV does not order held-out performance.** real_rep2's final CV is
  ~140 nats below real_rep1's and real_rep3's, yet its held-out gain (+56) is twice
  real_rep3's (+26).

| round | recovery_literal_rep1 | recovery_literal_rep2 | recovery_salience_rep1 | recovery_salience_rep2 |
|---|---|---|---|---|
| seeds | -18757.6 / 0 | -18755.3 / 0 | -17426.3 / 0 | -17426.9 / 0 |
| 0 | -18657.5 / +23.3 | -18659.9 / +23.7 | -17413.3 / +4.3 | -17411.4 / +7.8 |
| 1 | -18656.3 / +23.6 | -18658.4 / +23.3 | -17405.8 / +11.2 | -17410.5 / +7.7 |
| 2 | -18654.8 / +23.3 | -18656.3 / +23.3 | -17405.8 / +11.2 | -17395.2 / +13.1 |
| 3 | -18654.4 / +23.5 | -18655.5 / +23.6 | -17390.9 / +13.5 | -17393.2 / +13.4 |
| 4 (end) | -18654.4 / +23.5 | -18655.5 / +23.6 | -17390.9 / +13.5 | -17393.2 / +13.4 |

The recovery cells reach the ground truth's held-out level (+23.8 for literal, +13.5 for
salience, vs the best seed) by round 0-3.

## Did CV selection export models that generalise?

| | rep 1 | rep 2 | rep 3 |
|---|---|---|---|
| run 1 (trial-level PSIS-LOO), exported held-out rank | 26/40 (-7.5, SE 71.3) | 4/40 (+83.2, SE 27.4) | - |
| run 2 (grouped CV), exported held-out rank | **2/58** (+84.3) | **8/57** (+56.2) | **11/58** (+26.4) |
| run 2: held-out ranks of the 12 live models | 1, 2, 6, 7, 9, 11, 12, 15-19 | 1-8, 11, 16, 18, 20 | 6, 7, 8, 10, 11, 13, 15, 17, 19-21, 24 |

- **Better than run 1, with no collapse.**
  - real_rep1's and real_rep2's live sets hold the top held-out models (ranks 1-2 and
    1-8).
  - The live sets stayed at the cap of 12; run 1 ended with 4 and 1.
- **real_rep3 is the weak case.** Its top five held-out models were all pruned:
  - `valence_salience_listener` +51.5 (SE 36.2) and `evaluative_elaboration_listener`
    +49.7 (SE 35.9) are not distinguishable from the winner;
  - `feature_surprisal_listener` +38.5 (SE 18.1) is ~12 lpd ahead of the winner on
    held-out data. Grouped CV pruned it as 381.8 ELPD-CV behind (> 2 x dse 133.0).
  - So CV over training conditions did not predict transfer to new conditions for this
    model (§5 item 9).

## Recovery without internet access

| cell | exported (RMSE to GT) | closest live (RMSE) | best seed (RMSE) | `seeds_would_pass` | exported - GT held-out | verdict |
|---|---|---|---|---|---|---|
| recovery_literal_rep1 | parsimonious_literal_mixture (0.0053) | graded_specificity_listener (0.0024) | rsa_l1_salience (0.0485) | false | -0.31 (SE 0.61) | recovered |
| recovery_literal_rep2 | distinctiveness_neutral_prior (0.0032) | distinctiveness_heuristic_listener (0.0027) | rsa_l1_shared_prior (0.048) | false | -0.26 (SE 0.55) | recovered |
| recovery_salience_rep1 | conservative_base_rate_simplicity (0.0012) | the same | rsa_l2 (0.049) | false | -0.03 (SE 0.09) | recovered |
| recovery_salience_rep2 | salience_base_rate_ambiguity_l1 (0.0030) | the same | rsa_l2 (0.049) | false | -0.05 (SE 0.05) | recovered |

- **All four recovered; recovery is at least as close as in run 1** (exported 0.004-0.009 there).
- **In both salience cells the exported model is also the closest live one.** Run 1's
  never was.
- **No starting model would pass** the verdict in any cell.
- **Agents found the ground truths' mechanisms without any web access**: no agent fetched
  anything (§3, network).

## 1. What ran

| | |
|---|---|
| Sweep root | `$SCRATCH/auto-psych/rsa_run2` |
| Code | `dec3790ebf8e0c1ce99db8536d1b21e5e1a2818c` (staged; data `prepared_code` too) |
| Jobs | prepare 46968137; setup 46968321; array 46968325 (tasks 0-6); resume of tasks 3-5: 46996637; held-out pages 47004423, 47014741, 47018866, 47020916 |
| Partition | `-p mcfrank` throughout; 4 CPUs / 30G / 24 h per cell; six at once, the seventh started when the first recovery cell ended |
| Settings | real x 3 x 8 rounds, recovery x 2 x 4 rounds, 6 slots; grouped 5-fold CV for selection and pruning; live cap 12; no agent network; per-source standings in briefs |

**Data.**
- `combined_trials.csv`, `real/train.csv` and `real/test.csv` hash exactly as in run 1.
  `real/split.json` differs only in its recorded `trials_csv` path (`rsa_run1c` → `rsa_run2`).
- Setup simulated both recovery datasets and checked their held-out units and rows
  against the real split.

### Wall time, memory, cost

| cell | wall | MaxRSS | agent runs | cost |
|---|---|---|---|---|
| real_rep1 | 10:17 | 22.2 GiB | 59 | $89.18 |
| real_rep2 | 12:09 | **26.3 GiB** | 62 | $93.69 |
| real_rep3 | 8:48 | 22.3 GiB | 54 | $72.95 |
| recovery_literal_rep1 | 2:02 + 5:30 (resumed) | 14.0 GiB | 40 | $66.19 |
| recovery_literal_rep2 | 1:27 + 5:08 (resumed) | 15.4 GiB | 36 | $53.96 |
| recovery_salience_rep1 | 1:50 + 3:18 (resumed) | 12.8 GiB | 36 | $41.93 |
| recovery_salience_rep2 | 4:26 | 16.1 GiB | 30 | $34.01 |
| **total** | 22:35 → 12:46 PDT | | **313** | **$451.89** |

- Costs are summed from each cell's `token_usage.jsonl`. `token_usage_summary.json`
  undercounts the resumed cells (§5 item 4).
- No call reported missing usage.
- The real cells used 74-88% of their 30 GiB.

## 2. Failures and what was done

1. **First submit went to the wrong `WORK_ROOT`.**
   - `submit.sh` (also `rsa_status.sh` and `rsa_loop_array.sbatch`) still defaults to
     `rsa_run1`; only `_env.sh` defaults to `rsa_run2`.
   - The data check passed because run 1's aborted `rsa_run1/` has a `train.csv`.
   - Cancelled 46968229/46968230 before they started; nothing was written.
   - Resubmitted with `export WORK_ROOT=$SCRATCH/auto-psych/rsa_run2`, which every later
     command also exports. Not committed (not blocking).
2. **Three recovery cells failed in round 2, just after midnight (00:03, 00:26, 00:37).**
   - The cells were recovery_literal_rep1, recovery_salience_rep1 and
     recovery_literal_rep2.
   - The error was `the fit process exited with code 1 without reporting`, from a
     candidate's spawned fit child.
   - The child's traceback is arviz's once-a-day marker: `os.replace` of
     `<agent tree>/.xdg/cache/arviz/daily_warning.tmp` → `FileNotFoundError`. A cell's
     concurrent fit children share one cache directory and collided on the fixed-name temp
     file.
   - This is the bug that killed 11 of 24 SR cells on 2026-09-28. SR fixed it with a
     cache directory per fit process (`_fit_process_caches`); the RSA loop's fit children
     do not have that.
   - The loop rightly called it infrastructure and stopped.
   - Every tree's marker already read 2026-10-08, so it could not recur before midnight. I
     resumed tasks 3-5 at 07:15 (`ARRAY=3-5 SKIP_SETUP=1`, job 46996637).
   - Each resumed from its last scored step. The half-done round 2 is kept as
     `round_2_abandoned_1` and was rerun: about $10-15 of agent work lost per cell.
   - Not committed (resume needed no code change).
3. **The handoff's git 1.8 pull recipe is a silent no-op.**
   - `git fetch origin auto-rsa` updates only `FETCH_HEAD` on git 1.8, so `git merge
     --ff-only origin/auto-rsa` reports "Already up-to-date".
   - Sherlock stayed at dec3790 when I first pulled ff61007. `git fetch origin` (no
     refspec), then the merge, works.
   - Run 2 started on the right code because I had run a plain fetch beforehand.
4. **Partial commits at the PI's request, against the run's no-commit rule:**
   - **29fa52c**: real_rep3 early;
   - **fb6a32d**: real_rep1, recovery_salience_rep2, and §7 pages for three cells;
   - this commit: the rest.

   Each holds data only; no code was committed.
5. **recovery_literal_rep1 was the last cell** (resumed 07:15, done 12:46). Its round 4 ran
   longest: 11 rejections, the most repairs. It was pushed in a follow-up commit to
   66cd001.

## 3. Checks (handoff §3, and over the whole run)

- **No network.** Every cell's loop verified the no-network shell on the node before its
  agents started; no `does not run here` / `left internet sockets open` errors.
  - Across all seven cells (313+ agent runs) no agent called `webfetch` or `websearch`.
  - `agent_activity.md` still lists "URLs in the agent logs". These are URLs that
    *appear* in log text, not fetches:
    - arviz's warning URL;
    - doc URLs in model output;
    - in recovery_salience_rep2, 84 `files.pythonhosted.org` URLs. I checked those: an
      agent's `grep` for `dist.` matched `uv.lock` lines in its read-only tree.
  - So the handoff's "should list no URLs" can't be read literally from this file
    (§5 item 8).
- **CV ran.** Every cell has `.cv/folds.json` with 5 folds (real cells: 39,955 training
  trials, folds of 7,803-8,407). Every standing row in `history.json` carries `elpd_cv`,
  `cv_diff`, `cv_dse` and `cv_behind_by_source`.
- **Per-source standings reached the agents.** Every round-2 `existing_hypotheses.md`
  has "by source: ..." lines.
- **Self-checks.** Every agent run but one logged PASS/FAIL lines from `check_candidate`
  (recovery_literal_rep1: 37 of 38).
- **Held-out pages (§7):** 0 refits in every job (`not in` never appears); 3-4 minutes
  per batch.
- **Leaks.**
  - No `gt_name_mentions.txt` in any cell.
  - Agent trees passed `check_agent_tree.sh`: withheld seeds absent, no `test.csv`,
    `split.json`, `provenance.json`, `combined_trials.csv` or `simulated.csv`.
- **Committed files:** scanned for API keys and Prolific IDs (none). The only CSVs are
  aggregate tables (`heldout.csv`, `unit_lpd.csv`: lpd sums and counts per unit).

## 4. Rejections

Mostly the novelty gate (predictions within RMSE 0.002 of an admitted model), plus a
few non-converged fits.

| cell | admitted / rejected / pruned |
|---|---|
| real_rep1 | 48 / 10 / 41 |
| real_rep2 | 47 / 11 / 40 |
| real_rep3 | 48 / 6 / 41 |
| recovery_literal_rep1 | 20 / 11 / 12 |
| recovery_literal_rep2 | 21 / 6 / 13 |
| recovery_salience_rep1 | 23 / 4 / 20 |
| recovery_salience_rep2 | 22 / 8 / 20 |

## 5. Notes for the driver session

1. **`WORK_ROOT` defaults:** `submit.sh:15`, `rsa_status.sh:18` and
   `rsa_loop_array.sbatch:47` still say `rsa_run1`. Make them match `_env.sh` (or take
   it from one place).
2. **git 1.8 pull recipe** (handoff §0, §7): use `git fetch origin && git merge --ff-only
   origin/auto-rsa`. `git fetch origin auto-rsa` leaves `origin/auto-rsa` stale on git 1.8.
3. **arviz midnight collision in the RSA loop:** give each fit child (`_fit_cached`'s
   spawned process, and any other spawned fit) its own `XDG_CACHE_HOME`, as SR's
   `_fit_process_caches` does. It cost three cells a round.
4. **`token_usage_summary.json` covers only the last process of a resumed cell.**
   - recovery_salience_rep1's summary: 22 calls, $24.84. Its `token_usage.jsonl`: 36
     calls, $41.93.
   - The overview page (`sweep_report`) reads the summary, so it understates resumed
     cells; summarise from the jsonl.
5. **`prepare.sbatch`** writes `%x_%j.out` into the checkout. I passed `-o` to scratch;
   default it to `$WORK_ROOT/logs`.
6. **`heldout_reports.sbatch`** expects the brought-back cells inside Sherlock's
   checkout, but the bring-back happens on the Mac. I built a scratch copy
   (`$WORK_ROOT/bring_back/<cell>`, same file set) and passed `BRING_BACK`. Untracked
   generated files in the checkout would collide with the same files committed from the
   Mac on the next pull. Make the scratch copy the documented path, or have the script
   build it.
7. **Memory.** The 8-round real cells peaked at 22-26 GiB of 30 GiB (real_rep2: 26.3).
   - Longer runs or a larger live cap need more. But 6 x 40G = 240 GiB exceeds the
     node's 192 GB, so that means fewer cells at once or 5 x 36G.
8. **`agent_activity.md`:** separate *fetched* URLs (web-tool calls) from URLs that
   merely appear in log text, so the no-network check is readable from it.
9. **CV over training conditions vs transfer to new conditions** (a finding, not a bug):
   - in real_rep3, grouped CV pruned `feature_surprisal_listener` 381.8 behind the
     winner, but it is ~12 lpd ahead on held-out conditions;
   - real_rep2's final CV is ~140 below the other replicates, yet its held-out gain is
     the second largest;
   - the gains sit almost entirely on pragmods and Sikos 2021.

   Worth checking whether CV within the training conditions mainly measures fit to
   Mayn & Demberg's many conditions (most of the training trials), while the held-out
   test weights pragmods heavily, as run 1's re-score suggested.
10. **Sherlock's `~/auto-psych` `main` was not pushed** (the requested `git push origin
    main:sherlock-main-2026-10-08`). It is pre-rewrite history:
    - 117 commits, mostly Ben Prystawski's (latest 2026-06-22);
    - 103 of its 104 non-merge commits have identical patches on GitHub's `main`, and
      every subject and every path is there;
    - the one difference (`bc1df3b` vs `3890a10`) is a Prolific-style worker ID the
      rewrite replaced with `"REDACTED"`.

    Pushing it would put the unredacted pre-rewrite history back on GitHub. Nothing is
    stranded. The ref can be reset to `origin/main` whenever the PI wants.

## 6. `rsa_status.sh` at the end

```
sweep: /scratch/users/mcfrank/auto-psych/rsa_run2   code: dec3790ebf8e0c1ce99db8536d1b21e5e1a2818c
task  cell                    state  stage   best model                              adm/rej/pruned  tokens  cost    agents  recovered  last exit
0     real_rep1               done   scored  rsa_l2_singleton_feat_color_valence_l0  48/10/41        551.0M  $89.18  59      -
1     real_rep2               done   scored  surprisal_isolated_graded_costly_l5     47/11/40        576.8M  $93.69  62      -
2     real_rep3               done   scored  crowd_discrim_confusion_chromatic_l2    48/6/41         415.9M  $72.95  54      -
3     recovery_literal_rep1   done   scored  parsimonious_literal_mixture            20/11/12        426.7M  $66.19  40      yes
4     recovery_salience_rep1  done   scored  conservative_base_rate_simplicity       23/4/20         274.1M  $41.93  36      yes
5     recovery_literal_rep2   done   scored  distinctiveness_neutral_prior           21/6/13         353.0M  $53.96  36      yes
6     recovery_salience_rep2  done   scored  salience_base_rate_ambiguity_l1         22/8/20         210.0M  $34.01  30      yes
```

Costs here are from each cell's `token_usage.jsonl`, so the resumed cells are complete.

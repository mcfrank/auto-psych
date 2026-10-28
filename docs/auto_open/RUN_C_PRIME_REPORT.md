# Run C′ (run C's fixes, checked)

From the local Sherlock session, 2026-10-28, for the driver session. Handoff: `HANDOFF_rehearsal.md`,
"Run C′". Code `a5a62604` for both cells (`code_commit`). Times are PDT.

**Both cells finished in 6–7 h, one experiment each, with no error and no restart.**

What worked:
- **The theorists now write candidates.**
  - 5 of 9 sessions wrote a `candidate.py`; in run C it was 4 of 23.
  - In the open cell, all 3 first attempts wrote one, at 16, 27 and 38 min.
  - 4 candidates were admitted, 2 per cell (§3).
- **Every fit finished well within its limit.** The slowest was 61 min, so the 4-h ceiling was
  never reached. There is no `dropped ... its fit failed` line in either log (§4).
- **Untrusted theories were dropped on record.** Four in the open cell, each with
  `untrusted, not carried` in `history.json` and the ledger (§4).
- **The designer submitted on its own, at 75 of 120 min.** The design passed its checks on the
  first try. No design fit was screened out.
- **Both pilots passed. Both simulated collections are complete. `gt_name_mentions.txt` is empty
  in both cells.**
- **Both loops moved toward the truth (§6).** Each new incumbent is closer to the truth on exp1 than
  every starting theory. Neither has reached it: they are 3 SE (fixed) and 5 SE (open) behind.

What fell short:
- **The open (betting) paradigm broke cross-validation for the simple theories (§4).**
  - Three of the six starting theories could not be scored: `rsa_l1`, `rsa_l2` and
    `oddity_heuristic_listener`. So did one admitted theory, `competitor_confusion_l2`.
  - All four were dropped as untrusted, so the open cell ends without its vanilla RSA baselines.
  - The unconverged `rsa_l2` fit scored **+121,409 nats** on the betting data, against about
    −16,400 for the others. That is a density spike, not a fit.
- **The theorists still mostly run to the clock.** 7 of 9 sessions were stopped by the 60-min
  limit rather than finishing; 2 of them wrote nothing. Two candidates were first written at
  minutes 57 and 38, not by minute 15. Theorists still write their own fit code inline (§3).
- **The one repair resubmitted the rejected file unchanged** and was rejected for the same
  reason (§3).
- **Both critics needed a second attempt.** Both first attempts ran out of context ("length")
  after about 14 min, having written no statistic (§5).
- **Spend was $25.92, against the handoff's estimate of about $15** (§7). The pilot model is still
  unpriced.

## 1. Jobs and bring-back

| job | cell | partition | outcome |
|---|---|---|---|
| 49920786_0 | open | mcfrank | **COMPLETED**, 7 h 0 min (23:58 → 06:58) |
| 49920789_0 | fixed | owners | **COMPLETED**, 6 h 4 min (00:20 → 06:24); not preempted |
| 49965684 | ground-truth check: fixed | normal → mcfrank,normal | **COMPLETED**, 32 min |
| 49965679 → 49978048 | ground-truth check: open | normal → mcfrank,normal | 49965679 cancelled while pending; 49978048 **COMPLETED**, 1 h 6 min |

- **The ground-truth checks** were submitted to `normal` as the handoff says. They waited, so I
  let them start on `mcfrank` too (`open_oracle.sbatch`'s own default). They run with
  `--no-requeue`.
- **The fixed check used about 1.5 CPUs on average and 9.5 GB at peak, of the 24 CPUs and 96 GB it
  asks for.** A labmate's jobs held 20 of the lab node's CPUs, so I cancelled the open check while it
  was still pending. I resubmitted it with `--cpus-per-task=8 --mem=32G`, and it started at once.
  `open_oracle.sbatch`'s request could come down to about that.
- **Brought back** to `data/rsa_open/plumbing2/<cell>/outputs/`, as for run C:
  - each cell's outputs, with the `.fit_cache/*.json` sidecars force-added;
  - `ground_truth.json`, `code_commit`, each cell's `gt_name_mentions.txt` and `DONE`, and the two
    job logs in `logs/`.
- **Left on Sherlock:** corpora, E's trials, `.nc` fits, CV folds, agent scratch and homes, and
  `lists.json`. Each design's `record.json` holds the `lists.json` sha256.
  - The fixed cell's `lists.json` is still 349 MB, and the open cell's is 52 MB. That is the whole
    design. The per-list live pages are built at deploy, so run D1 is their first test.

## 2. Per cell

| | open | fixed |
|---|---|---|
| design | "Graded Pragmatic Belief Allocation Across Competitor Structures": bet, 150 people × 24 trials | "Which one does Bob mean? (forced choice)": choice, 400 × 13 |
| submitted by | designer, 75 of 120 min (00:18 → 01:32), one cycle; frozen 01:59 | fixed arm; frozen 01:21 |
| joint EIG | 3.73 bits; 0.0186 bits/$ (floor 0.0064) | 3.63 bits; 0.0136 bits/$ |
| pilot | comprehension 1.0, catch 1.0, readable 1.0; 6.0 min | 1.0, 1.0, 1.0; 2.3 min |
| design fits | 15, none screened out | 15, none screened out |
| collection | 150 × 27 rows (24 trials + 3) | 400 × 16 rows (13 + 3) |
| admission limit | 30 min (2 × the median starting fit) | 33 min |
| theorist sessions | 4 (3 + 1 repair) | 5 (3 + 2 retries) |
| admitted | 2 | 2 |
| exported best | `extension_contrast_l2` (new) | `rsa_l2_shepard_distinct_valence_l0` (new) |
| live set | `extension_contrast_l2`, `rsa_l2_singleton_feat_color_valence_l0` | `rsa_l2_shepard_distinct_valence_l0`, `contrastive_prototype_euclidean_l2` |

**Both cells changed incumbent to a discovered theory.** In each, the other carried theory is kept as
the best on E (the guard's reference there).
- **Open:** `extension_contrast_l2` leads on exp1 by 135.9 ± 37.3 nats. It is 85.7 ± 50.4 behind
  `rsa_l2_singleton_feat_color_valence_l0` on E.
- **Fixed:** `rsa_l2_shepard_distinct_valence_l0` leads on exp1 by 32.1 ± 16.3. It is 25.4 ± 9.5
  behind `contrastive_prototype_euclidean_l2` on E.

## 3. The theorists (Gemini 4, `google/gemini-4-argon` via opencode)

| session | role | candidate | first `candidate.py` | `--quick` / full checks | ended by | outcome |
|---|---|---|---|---|---|---|
| open 1 | explore | yes | 15.6 min | 2 / 0 | clock (56.8 min) | **admitted** `extension_contrast_l2` |
| open 2 | refine incumbent | yes | 27.1 min | 2 / 1 | itself (40.5 min) | rejected: fit over 30 min |
| open 2, repair | | same file | — (unchanged) | 2 / 0 | clock (56.4 min) | rejected again: fit over 30 min |
| open 3 | refine chosen | yes | 38.3 min | 1 / 1 | itself (47.0 min) | **admitted** `competitor_confusion_l2`, then untrusted (§4) |
| fixed 1 | explore | yes | 57.0 min | 1 / 0 | clock (59.7 min) | **admitted** `contrastive_prototype_euclidean_l2` |
| fixed 2 | refine incumbent | no | — | 0 / 0 | clock (59.4 min) | no file, retried |
| fixed 2, retry | | yes | 10.4 min | 1 / 1 | clock (55.5 min) | **admitted** `rsa_l2_shepard_distinct_valence_l0` |
| fixed 3 | refine chosen | no | — | 0 / 0 | clock (59.0 min) | no file, retried |
| fixed 3, retry | | no | — | 0 / 0 | clock (58.8 min) | no file |

"First `candidate.py`" is the first write of the file in the session's transcript (`agent.jsonl`).
"Ended by clock" means the session's last step ended in `tool-calls`, not `stop`.

- **5 of 9 sessions wrote a candidate.** Two of those five wrote it by minute 15 (15.6 and 10.4
  min). The brief asks for a first version within 15 min, and three were later, at 27, 38 and 57
  min.
- **The self-checks were used.** 7 of 9 sessions ran `check_candidate`: 12 `--quick` and 4 full
  runs in all. The three sessions that never ran one wrote nothing.
- **A fit script of its own: yes, inline.** No session wrote a `.py` file besides `candidate.py`.
  But they ran 21–49 inline `python -c` scripts each against the corpus, and some of those are
  fits:
  - **MAP fits with `scipy.optimize.minimize`** in open 1 and fixed 3's retry;
  - **numpyro `MCMC`/`NUTS`** in open 2's repair (9 runs) and fixed 3's retry (2 runs);
  - the rest score `choice_probs` or read the cached `.nc` fits.
- **The repair did not repair.**
  - The rejection reason was a fit over 30 min, "still in warmup, adapting its step size".
  - The repair session spent its hour on 9 NUTS runs of its own and never edited `candidate.py`.
  - The harness then fitted the copied-in file again, as `rsa_l2_confusion_valence_l0_2`, and
    rejected it for the same reason. This is a refinement of the incumbent: one new parameter on
    `rsa_l2_singleton_feat_color_valence_l0`.
- **Compared with run C:** 5 of 9 sessions wrote a candidate, against 4 of 23. Every candidate that
  fitted within the admission limit was admitted.

## 4. Fits, cross-validation and untrusted theories

**Fit times** come from the `.fit_cache` sidecars, which cover full fits and folds:
- open: 1.7–25.8 min;
- fixed: 3.7–61.0 min. The slowest is `isolated_graded_costly_l3`'s full fit.

Nothing came near the 4-h ceiling (`IN_PLAY_FIT_LIMIT_SEC`), and neither log has a
`dropped ... its fit failed` line.

**The fold limit is what bit, in the open cell only.** Each fold is limited to 3 × the theory's own
full-fit time (`FOLD_TIME_FACTOR`), with a 10-min minimum. Four theories were left unscored and then
dropped as `untrusted, not carried`:

| theory | why untrusted |
|---|---|
| `rsa_l2` (starting) | fold 2: 993 of 2,000 transitions divergent, R-hat 2.48, bulk ESS 5 |
| `oddity_heuristic_listener` (starting) | a fold did not finish within 28 min, still in warmup |
| `rsa_l1` (starting) | a fold did not finish within 18 min, still in warmup |
| `competitor_confusion_l2` (admitted) | a fold did not finish within 31 min, still in warmup |

- **The drop rule works as designed.** Each event is in `history.json` step 2 and in the ledger, and
  the incumbent is trusted.
- **On the betting data the plain theories' folds get stuck.** Their full fits took 3–10 min. On the
  fixed cell's forced-choice data, every theory's folds converged (`cv_converged: true` for all
  eight).
  - So the open cell's final set has no vanilla RSA baseline.
  - The prune rule never compared those theories to the incumbent: it left them unscored, not
    beaten.
- **The +121,409-nat ELPD.** At step 0, `rsa_l2`'s exp1 `elpd_cv` was **+121,409**. Every other
  theory scored −16,400 to −17,000.
  - The bet measure is a continuous allocation, so its density can exceed 1. The unconverged fold
    appears to have found a spike.
  - The rule kept it out, because untrusted theories are not ranked. But at step 0 every other
    theory's exp1 "behind" was computed against it: about 138,000 ± 39,800 nats.
  - By step 2 `rsa_l2` was gone and the comparisons were sensible again.
  - **I think the betting likelihood needs a look** before a betting design is run live: a bound on
    the density, or a check for a degenerate fold. That is your call.
- **Fixed cell: no theory was untrusted.** Three starting theories were pruned as ineligible
  (`rsa_l1`, `rsa_l2`, `oddity_heuristic_listener`): more than 4 SE behind the best on E. Three more
  were more than 2 × the clustered dse behind the new incumbent on exp1.

## 5. The critic

| cell | attempt 1 | attempt 2 | statistics | significant (FDR) |
|---|---|---|---|---|
| open | 13.7 min, ended `length`, no statistic | 7.8 min, `stop` | 8 evaluated | 3 (3) |
| fixed | 14.5 min, ended `length`, no statistic | 14.3 min, `stop` | 8 evaluated | 4 (3) |

- **Both first attempts ran out of context, not time.** Their last step ended with finish reason
  `length` after 11–13 tool steps.
  - The open critic had been reading the design's theory files and `design.json` 500–2,000 lines at
    a time.
  - The fixed critic was reading inline-script output over the 52 scenes.
- **The retries succeeded**, so every round was critiqued: 8 of 8 statistics evaluated in both
  cells, against 4/8, none, 0/8 and 4/8 in run C.
- **A cheap fix might be** to keep `design.json` and the design's theory folder out of the critic's
  reach, or to tell it not to read them whole. Your call.

## 6. Ground-truth checks

Values are ELPD-CV minus the truth's, ± the clustered SE (negative: worse than the truth). The
truth generated exp1; E is real people (pragmods). The outputs are in `private/oracle_<cell>_exp1/`
on Sherlock, and `oracle.md` is copied to `data/rsa_open/plumbing2/oracle/`.

**Fixed cell** (49965684, 32 min):

| theory | ended | E − truth | exp1 − truth |
|---|---|---|---|
| **rsa_l2_shepard_distinct_valence_l0** (incumbent) | live | +96.5 ± 56.8 | **−29.5 ± 9.5** |
| contrastive_prototype_euclidean_l2 | live | +121.9 ± 56.5 | −61.7 ± 18.8 |
| rsa_l2_singleton_feat_color_valence_l0 | pruned | +102.5 ± 56.5 | −85.1 ± 19.9 |
| isolated_graded_costly_l3 | pruned | −53.0 ± 27.1 | −103.5 ± 26.3 |
| competitor_confusion_speaker | pruned | −88.9 ± 41.3 | −296.7 ± 67.5 |
| oddity_heuristic_listener | pruned | −618.7 ± 138.1 | −684.3 ± 103.5 |
| rsa_l1 | pruned | −167.7 ± 57.6 | −824.5 ± 131.6 |
| rsa_l2 | pruned | −151.5 ± 53.3 | −850.4 ± 132.2 |

- **The loop moved toward the truth.** The new incumbent is closer to the truth on exp1 than every
  starting theory: −29.5 ± 9.5, against −85.1 for the best starting theory.
- **It has not reached the truth.** It is still about 3 SE behind on exp1, after one round of 3
  theorists. In run C, the fixed-pair cell's incumbent tied the truth.
- **The ordering on exp1 matches the loop's own.** The prune decisions agree with the truth's
  ranking.
- **On E, the `rsa_l2_*` theories beat the truth by about 100 nats** (1.7–2.2 SE), as in runs B′ and
  C. The simulated truth is not the best account of real people.

**Open cell** (49978048, 1 h 6 min):

| theory | ended | E − truth | exp1 − truth |
|---|---|---|---|
| **extension_contrast_l2** (incumbent) | live | +7.7 ± 16.6 | **−82.7 ± 16.5** |
| rsa_l2_singleton_feat_color_valence_l0 | live | +93.4 ± 57.7 | −218.6 ± 40.5 |
| isolated_graded_costly_l3 | pruned | −51.6 ± 27.8 | −397.4 ± 84.7 |
| competitor_confusion_speaker | pruned | −101.8 ± 42.5 | −788.2 ± 159.1 |
| rsa_l2 | untrusted, dropped | −156.8 ± 50.1 | **+137,590.6 ± 39,762.7** |
| competitor_confusion_l2, oddity_heuristic_listener, rsa_l1 | untrusted, dropped | — | — |

The truth's own exp1 ELPD-CV is −16,181.2. "—" means the theory has no score: its folds failed in
the loop, as §4 describes.

- **The new incumbent is closer to the truth than any starting theory,** by about 136 nats on exp1
  (−82.7 against −218.6). On E it is within 1 SE of the truth (+7.7 ± 16.6).
- **It is still 5 SE behind the truth on exp1.**
- **The truth fits the betting data normally.** Its folds converged, and its ELPD-CV is in line with
  the trusted theories'.
  - So the +137,591 is `rsa_l2`'s degenerate fold, not the likelihood misbehaving for a sound fit.
  - Even so, an unconverged fit can produce a score that large, and the same hazard applies to
    every betting design. Only the trust check keeps it out of the ranking.
- The check's log has one `invalid value encountered in sqrt`, probably from the SE of `rsa_l2`'s
  degenerate score.

## 7. Spend (`token_usage.jsonl` per cell)

| cell | theorists | critic | designer | pilot | total |
|---|---|---|---|---|---|
| open | $8.11 (4) | $1.64 (2) | $1.33 (1) | unpriced (149) | $11.08 |
| fixed | $12.69 (5) | $2.15 (2) | — | unpriced (72) | $14.84 |
| **total** | $20.80 | $3.79 | $1.33 | | **$25.92** |

Record counts are in parentheses. Model IDs: theorists, critic and designer are
`google/gemini-4-argon`; the pilot is `gemini-3.8-flash`.

- **`token_usage_estimates.json`:** `estimated` is empty, so every Gemini 4 record was priced this
  time. `unpriced_models` lists only `gemini-3.8-flash`:
  - open: 149 records, 466,492 tokens;
  - fixed: 72 records, 186,076 tokens.
  
  It has no priced record to estimate from, as `spend.py` says.
- **Against the handoff's $15:** the theorists are 80% of the spend. Sessions that run to the
  60-min clock cost about $2–3 each.

## 8. Issues for you

1. **Betting-paradigm cross-validation (§4).** Plain theories' folds get stuck or diverge, and one
   scored +121,409 nats. The open cell loses its RSA baselines. This matters for any open-arm
   design that picks the bet measure.
2. **Theorists still run to the clock (§3).** 7 of 9 sessions ended on it, 3 of 5 candidates came
   after minute 15, and they write their own fits inline: MAP and NUTS.
3. **A repair that resubmits the unchanged file (§3).** The harness could refuse an unchanged repair
   before fitting it again.
4. **The critic's first attempt runs out of context (§5),** in both cells.
5. **Spend is about 1.7 × the estimate (§7).**
6. **The per-list page is not tested yet (§1).** The design files are unchanged in size, and run D1
   is the first build of the new pages.
7. **Resource requests (§1, §9).**
   - `open_oracle.sbatch` asks for 24 CPUs and 96 GB, and uses about 1.5 CPUs and 10 GB.
   - `open_live.sbatch` asks for the same 24 CPUs and 96 GB, on `normal` by default.
   - Both wait for a whole node when they need much less.

## 9. Next

Run D fields these two experiment-1 designs. Both are frozen:
- open: `design/frozen`, 150 people, bet;
- fixed: `design/frozen`, 400 people, forced choice.

**D1 (test deploys, no money) was launched at 09:35 on the PI's word.** The jobs are 49980054
(open) and 49980055 (fixed), in work root `rsa_open_smoke`. They go to sites
`auto-psych-2c5da-ro-{open,fixed}-c0s`.
- The launcher submits to `normal` by default, and the handoff's command names no partition. The
  jobs waited there (estimated start 17:37), so at 10:20 I moved them to `owners,normal`.
- Each asks for 24 CPUs and 96 GB for a deploy.
- D1's results will be in a separate note.

D2 recruits real people. It needs the PI's explicit go-ahead, a Prolific balance check, and the PI's
review of the open (betting) paradigm (§4).

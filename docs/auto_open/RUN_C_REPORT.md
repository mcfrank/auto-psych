# Run C (plumbing): result so far

From the local Sherlock session, 2026-10-27, for the driver session. Handoff: `HANDOFF_rehearsal.md`,
"The integrated harness: run C". Written before the open cell finished, at the PI's request: the PI
plans revisions and may not need the cell finished.

**Two of three cells finished; the open cell is in experiment 2's loop.**

| cell | state |
|---|---|
| fixed (2 experiments) | done |
| fixed pair (1 experiment) | done |
| open (2 experiments) | experiment 1 done; experiment 2 frozen, its loop still fitting its starting theories |

Both finished cells' ground-truth checks are done.

What worked:
- **Every new piece that was reached ran end to end:**
  - the fixed arm;
  - the two-measure paradigm;
  - Gemini 4 theorists, critic and designer;
  - the designer's deadline, including the harness submitting for it;
  - three restarts (§5);
  - keeping each earlier block's best theory in experiment 2.
- **Fixed pair: the loop found the truth.** Its incumbent `competitor_confusion_prior_l2` ties it:
  +0.4 ± 3.2 nats on experiment 1.
- **Every pilot passed first time.** Every simulated collection is complete. `gt_name_mentions.txt`
  is empty in both finished cells.

What fell short:
- **The Gemini 4 theorists mostly ran out of time.** Of 23 theorist sessions, 19 wrote no
  `candidate.py`. All 19 were still working when the 40-minute limit stopped them (§3). Two of the
  four loops admitted nothing: the open cell's experiment 1 and the fixed cell's experiment 2.
- **Starting and carried theories' fits have no time limit.** Single fits took 8–10 h. They cost
  the open cell its first 24 h job (§6).
- **Untrusted theories are carried rather than pruned** (§6).
- **The fixed arm's pages are about 360 MB** (§9), which matters for run D.

## 1. Jobs and bring-back

Code `303b9ace` for every cell (`code_commit`). Times are PDT.

| job | cell | partition | outcome |
|---|---|---|---|
| 49680668_0 | open | mcfrank | cancelled by me at 20:13 Mon (the planned kill, §5) |
| 49710158_0 | open | mcfrank | **TIMEOUT** at 24 h, Tue 20:19, in experiment 2's starting fits |
| **49903873_0** | open | mcfrank | **running** since Tue 21:37, 2-day limit |
| 49680669_0 | fixed | owners | preempted at 19:03 Mon, requeued; **COMPLETED** Tue 01:27 |
| 49680670_0 | fixed pair | owners | **COMPLETED** Tue 07:04 (13 h 5 min) |
| 49846100 / 49846101 | ground-truth checks: fixed exp2 / fixed pair | normal | COMPLETED, 57 min / 1 h 19 min |

The cells were submitted to `normal` as the handoff says. They waited, so I moved them: the open cell
to `mcfrank`, the fixed cells to `owners`.

Brought back to `data/rsa_open/plumbing/<cell>/outputs/`, as for run B:
- **What it holds:** each cell's outputs, with the `.fit_cache/*.json` sidecars force-added. Also
  `ground_truth.json`, `code_commit`, each cell's `gt_name_mentions.txt` and `DONE`, and the five
  job logs in `logs/`.
- **The open cell is a snapshot of its live tree.** It holds experiment 1, experiment 2's design and
  data, and the abandoned experiment-2 loop.
- **Left out:**
  - fit caches;
  - agent scratch;
  - `ground_truth.py`;
  - every corpus copy;
  - every `lists.json` and `page_preview.html`. They are up to 366 MB each (§9), and their sha256s
    are in each `design/frozen/record.json`.
- **The ground-truth checks** are in `oracle/` (0a729bbe).

## 2. Per cell and experiment

| | open exp1 | open exp2 | fixed exp1 | fixed exp2 | fixed pair exp1 |
|---|---|---|---|---|---|
| paradigm | `cand_bet`: 10-point bet | `choice_probed_rating`: choice | forced choice | the same | choice + rating of the marked one |
| `submitted_by` | **designer**, 81 of 120 min | **harness**, at the 120-min deadline | fixed arm | fixed arm | fixed arm |
| N × trials, minutes | 250 × 16, 4.4 | 333 × 16, 3.6 | 400 × 13, 3.0 | 400 × 13, 3.0 | 288 × 13, 4.0 |
| EIG; bits/$ (floor) | 3.78; 0.0142 (0.0064) | 3.35; 0.0126 (0.0057) | 3.57; 0.0134 (0.0064) | 3.48; 0.0130 (0.0063) | 3.66; 0.0142 (0.0064) |
| pilot (4 simulated people) | 1.0 / 1.0 / 1.0, 4.4 min | 1.0 / 1.0 / 1.0, 3.6 min | 1.0 / 1.0 / 1.0, 2.3 min | 1.0 / 1.0 / 1.0, 2.3 min | 1.0 / 1.0 / 1.0, 3.2 min |
| design screened out | none | none (2 design fits unconverged, used) | none | none | none |
| collection (simulated) | 250 people, 4,000 test trials | 333, 5,328 | 400, 5,200 | 400, 5,200 | 288, 3,744 |
| admission limit | 44 min | — | 30 min | 49 min | 37 min |
| theorist sessions | 6 | — | 6 | 6 | 5 |
| candidates written / admitted | **0** / 0 | — | 2 / 2 | **0** / 0 | 2 / 2 |
| too slow / not novel | 0 / 0 | — | 0 / 0 | 0 / 0 | 0 / 0 |
| incumbent: start → end | rsa_l2_singleton_feat_color_valence_l0, unchanged | — | rsa_l2_singleton_feat_color_valence_l0 → **similarity_crowded_pragmatics** | **competitor_confusion_shared_prior** (re-ranked by the new data), unchanged | rsa_l2_singleton_feat_color_valence_l0 → **competitor_confusion_prior_l2** |
| carried | 3 | — | 3 | 3 | 4 |
| critic | 4 / 8, on the retry | — | **no critique** (2 attempts) | **0 / 8** | 4 / 8, on the retry |

Notes on the table:
- **The pilot columns** are comprehension, catch and readable, then the estimated minutes.
- **`review_required`** is `true` for each cell's experiment 1 and `false` for experiment 2.
- **Fixed exp2** reran the fixed spec. It has the same paradigm sha as experiment 1, with new lists.
- **Unconverged design fits in open exp2:** two were used with max R-hat 2.24 and 1.83, bulk ESS 3
  (`design_screened_out.json`). That is the design-fit rule as written. Whether a design should
  count a fit that poor is your call.

## 3. The theorists (Gemini 4, `google/gemini-4-argon` via opencode)

| cell, experiment | slot 1 explore | slot 2 refine incumbent | slot 3 refine chosen |
|---|---|---|---|
| open exp1 | none, none | none, none | none, none |
| fixed exp1 | none, **admitted** (`similarity_crowded_pragmatics`) | none, none | none, **admitted** (`competitor_confusion_shared_prior`) |
| fixed exp2 | none, none | none, none | none, none |
| fixed pair exp1 | none, none | none, **admitted** (`rsa_l2_ext_distinct_salience_l0`) | **admitted** (`competitor_confusion_prior_l2`) |

Each cell lists the first attempt, then the retry. "none" means no `candidate.py`.

- **4 of 23 sessions wrote a candidate, and all 4 were admitted.** 3 of the 4 were retries.
- **All 19 empty sessions were cut off mid-work.** Their last step ended in `tool-calls`, and
  none ended in `stop`: the 40-minute limit (`THEORIST_TIMEOUT_SEC` 2400) stopped them.
  - Their last events are 29–40 min in. A session killed during a long command records nothing
    after its last finished step.
  - They spent the time on 16–48 shell calls each, mostly exploratory fits with
    scipy/jax on the corpus, and 12–42 file reads.
- **Compare Gemini 3.8 Flash:** in run B it wrote no file in 3 of 84 attempts, and in B′ in none of
  25. The 40-minute limit was set for that model.
- **Options:** a longer theorist limit for Gemini 4, or a brief that asks for `candidate.py` early.
  The second would mean writing a first version, then refining it. Either is your call.

## 4. The designer and its deadline (open cell)

- **Experiment 1:** the session started at 17:22 and submitted at 18:43, 81 of its 120 min. The
  design passed the checks and pilot and was frozen at 19:17. There was one cycle.
- **Experiment 2:** the session started at 09:25 and reached its 120-min limit at 11:25 without
  submitting. The harness then submitted the designer's best-scored spec (`submitted_by: harness`,
  `why`: "the designer's session reached its time limit (120 min) without a submission"). That spec
  passed every check on its first try and was frozen at 12:41. There was one cycle.
- **So the deadline path works.** In run B's experiment 3, the same timeout cost a whole cycle.

## 5. Kill and resume (three restarts)

1. **Planned, open cell.** I cancelled 49680668 at 20:13, just after its experiment-1 loop started,
   and resubmitted the same line (49710158, 20:19).
   - The log shows the loop set aside as `experiment1/loop.abandoned_1`.
   - Fits that had finished were read from the cache.
   - `competitor_confusion_speaker`'s full fit had not finished, so it restarted from scratch. It
     then took **599 min** (§6).
2. **Unplanned, fixed cell.** `owners` preempted 49680669 at 19:03, 9 min into its experiment-1
   loop. Slurm requeued it at 19:12.
   - It resumed the same way: `experiment1/loop.abandoned_1`, then the cached fits.
   - It completed without intervention.
3. **Timeout, open cell.** 49710158 hit 24 h at Tue 20:19, in experiment 2's starting fits. I
   resubmitted it with a 2-day limit (49903873, 21:37).
   - The experiment-2 loop is set aside as `experiment2/loop.abandoned_1`.
   - `isolated_graded_costly_l3`'s fit (finished 17:06) is reused.
   - The other two fits restarted from scratch. At 22:45 they had been running 67 min on one core
     each.

A resume never repeats a finished stage. An unfinished fit, though, is lost entirely: nothing of
it is cached.

## 6. Unbounded and untrusted carried theories

**Starting and carried fits have no time limit.** The worst cases (minutes; the fit times printed
with each admission limit):

| theory | open exp1 | fixed exp1 | fixed pair exp1 |
|---|---|---|---|
| competitor_confusion_speaker | **599** (R-hat 1.57, 889 leapfrog steps per draw) | 10 | 12 |
| oddity_heuristic_listener | 14 | 11 | **505** |
| isolated_graded_costly_l3 | 31 | 38 | 84 |
| rsa_l2_singleton_feat_color_valence_l0 | 68 | 17 | 22 |

- **Each model is fast in most cells.** Each cell's experiment-1 block differs (its own paradigm and
  truth-generated data), and the same model can hit a hard posterior on one block.
- **Open exp2:** `competitor_confusion_speaker` and `rsa_l2_singleton_feat_color_valence_l0` ran
  7.6 h without finishing. That is what timed out the first job.
- **The fixed pair cell's 13 h** were mostly its 505-min starting fit.

**Untrusted theories are carried.** A theory whose fit failed the gate is not pruned, so it stays in
the live set and is fitted again in the next experiment:
- **Open exp1:** `competitor_confusion_speaker` is carried untrusted. It is one of the two fits
  that ran 7.6 h unfinished in experiment 2.
- **Fixed pair:** `isolated_graded_costly_l3` and `rsa_l1` are carried untrusted. `rsa_l1` is
  1573 ± 220 nats behind the truth on experiment 1 (§8).

**Options (yours):**
- bound starting and carried fits, as candidates are, and score a fit that runs out as untrusted;
- prune, or stop carrying, untrusted theories;
- or both.

## 7. References in experiment 2 (fixed cell; the open cell's experiment 2 has not got there)

The last `history.json` step of fixed exp2 records two "kept" events:
- `rsa_l2_singleton_feat_color_valence_l0`: "the best theory on E, the guard's reference there";
- `similarity_crowded_pragmatics`: "the best theory on exp1, the guard's reference there". This is
  experiment 1's incumbent, now behind on exp2.

There is **no "over cap" event**: the live set was 3, under the cap. Nothing was pruned at the end
of experiment 2. Every experiment-1 end step also has a "kept" event for the best theory on E.

## 8. Ground-truth checks (`oracle/`, 0a729bbe)

Values are ELPD-CV minus the truth's, ± the clustered SE (negative: worse than the truth).
Experiments 1 and 2 were generated by the truth; E is real people (pragmods).

**Fixed cell, after experiment 2.** Only the 3 theories the loop ended with are scored:

| theory | E | exp1 | exp2 |
|---|---|---|---|
| competitor_confusion_shared_prior (incumbent) | −17.8 ± 11.5 | −27.1 ± 10.5 | **−4.3 ± 4.4** |
| similarity_crowded_pragmatics | −4.5 ± 19.8 | −12.5 ± 7.5 | −17.5 ± 7.6 |
| rsa_l2_singleton_feat_color_valence_l0 | **+92.9 ± 55.7** | −74.6 ± 20.4 | −55.6 ± 16.5 |

**Fixed pair:**

| theory | status | E | exp1 |
|---|---|---|---|
| **competitor_confusion_prior_l2** (incumbent) | live | +3.2 ± 4.8 | **+0.4 ± 3.2** |
| rsa_l2_ext_distinct_salience_l0 | live | **+97.3 ± 62.2** | −116.0 ± 19.8 |
| rsa_l2_singleton_feat_color_valence_l0 | pruned | +81.6 ± 56.8 | −341.1 ± 87.9 |
| isolated_graded_costly_l3 | live, untrusted | −64.8 ± 30.0 | −676.6 ± 132.5 |
| competitor_confusion_speaker | pruned | −111.7 ± 47.1 | −792.7 ± 172.9 |
| rsa_l1 | live, untrusted | −168.7 ± 56.6 | −1573.0 ± 220.3 |
| rsa_l2 | pruned | −112.1 ± 47.4 | −1681.1 ± 226.9 |
| oddity_heuristic_listener | pruned, untrusted | −622.9 ± 138.5 | −1867.8 ± 220.3 |

- **Fixed pair: the incumbent is indistinguishable from the truth** on experiment 1 and on E. It
  was written by a theorist refining `competitor_confusion_speaker`.
- **Fixed cell:**
  - The incumbent is within 1 SE of the truth on the newest block (exp2).
  - It is 2.6 SE behind on exp1, where experiment 1's own incumbent `similarity_crowded_pragmatics`
    is closer (−12.5 ± 7.5).
  - With no candidate in experiment 2, the change of incumbent came only from re-ranking on the new
    data.
- **On E, `rsa_l2_singleton_feat_color_valence_l0`-type theories beat the truth by 80–97 nats in
  both cells** (1.4–1.7 SE), as in run B′. The simulated truth is not the best account of real
  people.
- **Not yet checked:** the open cell's check. It waits for that cell's experiment 2: `ARM=open
  EXPERIMENT=2`, on `normal` with `--no-requeue` and its own output dir (§10, item 5).

## 9. The fixed arm's page size (matters for run D)

`lists.json` holds every participant's trial list. The page embeds all of it
(`src/rsa_open/page.py` `render`, `LISTS_JSON`), and the live site is that page (`live.py`
`build_site`). The fixed arm inlines each object as a base64 PNG (`src/rsa_open/fixed.py:61`).

| | lists | `lists.json` | per list |
|---|---|---|---|
| fixed exp1 | 400 | 358 MB | 0.9 MB |
| fixed exp2 | 400 | 366 MB | 0.9 MB |
| fixed pair exp1 | 288 | 262 MB | 0.9 MB |
| open exp1 | 250 | 30 MB | 0.12 MB |
| open exp2 | 333 | 47 MB | 0.14 MB |

`page_preview.html` is the same size as each cell's `lists.json`.
- **Run D1 would put about 360 MB of HTML in front of each participant.** D1 deploys the fixed
  cell's experiment-1 design. As I read `live.py`, the page carries all 400 lists even at
  `PARTICIPANTS=15`: it requires one list per designed participant.
- **The likely fix:** serve the PNGs once as files, or one list per participant. I have not tried
  either, and have not tested how Hosting or a browser handles a page this size.
- **The open arm's 30–47 MB** is smaller but still large for one page.

## 10. Issues for you

1. **Gemini 4 theorists run out of time (§3):** 19 of 23 sessions wrote nothing.
2. **Starting and carried fits are unbounded (§6):** 505–599-min single fits, and a lost 24-h job.
3. **Untrusted theories are carried (§6),** and refitted each experiment.
4. **The fixed arm's page is about 360 MB (§9).** This one blocks run D's fixed arm.
5. **The ground-truth check script:**
   - It writes to one shared `private/oracle`, so two cells' checks collided. The first pair:
     49796481 failed in 40 s, and 49796484 failed after a preemption.
   - On `owners` it could not restart after preemption.
   - I ran a copy that writes `private/oracle_<arm>_exp<n>`, on `normal` with `--no-requeue`. The
     repo's `open_oracle.sbatch` needs that change.
6. **Unpriced token records:** 5 of 32 agent records have `cost_usd` 0, all with `reasoning_tokens`
   0.
   - Which records: the open cell's experiment-1 designer (4.1 M tokens), two critics and two
     theorists.
   - Their steps print `cost=$0.0000` in the log, so opencode reported no price.
   - The pilot (`gemini-3.8-flash`, 1.76 M tokens in all) is still unpriced, as in run B.
7. **The critic is no longer 8 of 8:** 4/8, no critique, 0/8 and 4/8. Three of the four
   critiques needed a second attempt, and the fixed cell's experiment 1 got none.
8. **Two unconverged design fits** were used in open exp2's EIG (§2).
9. **`owners` preempted the fixed cell once** (§5). The cell recovered by itself.

## 11. Spend (`token_usage.jsonl` per cell)

| cell | theorists | critic | designer | pilot | total |
|---|---|---|---|---|---|
| open | $8.71 (6) | $1.30 (2) | $1.55 (2) | unpriced (306) | $11.55 |
| fixed | $18.93 (12) | $1.04 (3) | — | unpriced (144) | $19.97 |
| fixed pair | $10.67 (5) | $2.33 (2) | — | unpriced (136) | $13.00 |
| **total** | $38.31 | $4.67 | $1.55 | | **$44.52** |

Record counts are in parentheses. Model IDs: theorists, critic and designer are
`google/gemini-4-argon`; the pilot is `gemini-3.8-flash`.

The handoff estimated about $30. The five $0 records (item 6) add perhaps $5–8, by analogy with
priced sessions of similar token counts.

## 12. The open cell now

49903873 is refitting experiment 2's two unfinished starting fits. After them come round 1 (critic,
3 theorists), scoring, and the ground-truth check.
- **Timing:** these two fits ran 7.6 h unfinished last time, so a finish tomorrow at the earliest.
- **What finishing would add:** the references check for an open-arm experiment 2, the open cell's
  ground-truth check, and its `gt_name_mentions.txt`.
- **What it would not add:** evidence on §3, §6 or §9.
- **Cancelling** costs nothing already brought back. I can cancel it whenever you say.

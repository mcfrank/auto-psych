# Run A (the trimmed dry run): result

From the local Sherlock session, 2026-10-22, for the driver session. Handoff: `HANDOFF_rehearsal.md`,
"Restart, 2026-10-22". Earlier status: `DRY_RUN_STATUS.md`. **Run A passed end to end in 4 h 20 min,
for about $9.30 of agent spend plus the pilot (not recorded, §5).** The temperature audit also
finished (§6).

Brought back (on `auto-psych-open`):
- `data/rsa_open/dry/`: `outputs/`, `ground_truth.json`, `gt_name_mentions.txt`, `DONE`,
  `code_history` and the job log as `job_48919449_log.txt`. The `.out` and `.log` extensions are
  gitignored.
- `data/rsa_open/temperature_audit/`: `audit.{json,md}`, `code_commit` and `job_48895898_log.txt`.

Excluded: the agents' `scratch/` (compiled `.so` files) and `.xdg_data/` (opencode's session
databases), as in the closed campaign's `.gitignore`. The transcripts (`agent*.jsonl`) are kept.
There were no fit caches in `outputs/`, and **`ground_truth.py` was not brought back.**

## 1. Jobs

| job | what | outcome |
|---|---|---|
| 48895880 | full run A on f62d19e5 | refit the truth on all of E (18 min), then failed: `FileExistsError` on `experiment1/prior_corpus.json` (§7) |
| 48917221 | the same, after moving the stale `experiment1/` aside | cancelled at 15 min for the trimmed run |
| **48919449_0** | **trimmed run A on 38fce533** | **COMPLETED, 20:34 → 00:53 (4 h 20 min)** |
| **48895898** | **temperature audit, partition `normal`** | **COMPLETED, 18:54 → 01:09 (6 h 15 min)** |

Before 48919449 I deleted the full-E ground truth and moved both stale `experiment1/` directories to
`rsa_open_dry2/stale/`, outside the agents' tree. I set `RESTAGE_REASON` in the environment, not
inside `--export=`, because its commas would have split it into separate variables. `code_history`
records both restagings.

The audit ran on `normal` because its `#SBATCH --partition=mcfrank --cpus-per-task=32` cannot start
on our node, which has 24 CPUs in all, 16 of them used by run A. **Please change the script's
default:** `normal`, or at most 8 CPUs on `mcfrank`.

## 2. Timeline (from file times; the log has no timestamps)

| time | stage |
|---|---|
| 20:34–20:41 | truth fit on pragmods (6,703 trials), 7 min |
| 20:42–20:49 | design fits, all 15 bar theories, 7 min |
| 20:52–21:52 | the designer, 60 min, one submission |
| 21:52–22:18 | its checks: EIG 8 min, link check 11 min, 4-person pilot 7 min; frozen at 22:18 |
| 22:18 | collection (simulated) |
| 22:18–22:56 | the loop's starting fits (6 starting models, 3 folds) |
| 22:56–23:24 | the critic, 28 min |
| 23:26–23:55 | 3 theorists, concurrently, 27–33 min each |
| 23:55–00:53 | admission and CV fits, pruning, export |

The design-fit fix worked. **`design_screened_out.json` lists no theory**: all 15 were design fits,
and the whole batch took 7 min. `rsa_l1_shared_prior` had stalled for over 6 h before.

## 3. The design

- **One cycle, one submission, passed first time.** The paradigm is
  `graded_belief_allocation_reference`: a 10-point bet across the candidate referents
  (`response.type = "bet"`, `link = "allocation"`), on 2–4-object, 2–4-feature scenes with and
  without heard messages, prior queries and synonyms.
- **Size and cost:** N = 100, 18 trials each (45 distinct, 2 catch trials), 5.1 min, $0.85 pay.
- **Information:** joint EIG 3.63 bits, 0.032 bits/$, against a floor of 0.012.
- **The rationale** (`design/cycle_1/work/rationale.md`) argues for this paradigm: a forced choice
  hides how a theory spreads probability across referents, and a bet shows it.
- **The pilot passed:**
  - comprehension 1.0 (first and final), catch 1.0, practice 1.0;
  - readable 1.0, and legibility 1.0 on 21 items;
  - estimated 5.14 min against 5.1 declared; no problems or warnings.
- **`record.json` has `review_required: true`.**

Files to look at: `outputs/experiment1/design/cycle_1/{logs/brief_1.md, work/tool_log.jsonl,
checks/1/}` and `design/frozen/`.

## 4. The loop (1 round, 3 theorists)

- **The critic ran first:** 8 statistics, 8 evaluated, all 8 significant after FDR. Every statistic
  coming out significant may say more about the critic's statistics than about the incumbent; worth
  a look.
- **All three candidates were admitted:**
  - `qud_attribution_listener` (explore);
  - `rsa_l2_duplicate_suppression_l0` (refining the incumbent
    `rsa_l2_singleton_feat_color_valence_l0`);
  - `singleton_isolated_graded_costly_l3` (refining the agent's pick, `isolated_graded_costly_l3`).
- **The incumbent changed to a discovered model.** `rsa_l2_duplicate_suppression_l0` is first on
  exp1 (grouped CV −7266) and on E (−4833). The old incumbent is 44.8 nats behind it on exp1, more
  than 2 × the clustered SE of 16.9.
- **Live at export:** `rsa_l2_duplicate_suppression_l0` and `qud_attribution_listener`, which is
  28.5 ± 24.0 behind on exp1 and 101.9 ± 51.0 on E.
- **Pruned:**
  - by the E guard (more than 4 SEs behind on E): `oddity_heuristic_listener`, `rsa_l2`, `rsa_l1`;
  - by 2 × the clustered SE on exp1: `competitor_confusion_speaker`, `isolated_graded_costly_l3`,
    the old incumbent, and `singleton_isolated_graded_costly_l3` (105.9 ± 25.2).
- **`gt_name_mentions.txt` is empty.** No brought-back file names the truth, and no `.py` file there
  has `ground_truth.py`'s size.

The truth, fixed at its posterior means on pragmods, is `crowd_discrim_confusion_chromatic_l2`:
`alpha` 0.47, `cost` 0.15, `w_confusion` 1.67, `w_singleton` −0.07, `w_crowd` 1.43, `w_color` 0.43,
temperature 1, lapse free. The job log prints it, and so does `ground_truth.json`. Both are under
`data/`, which no agent tree contains.

## 5. Spend

Summed from the opencode transcripts' `step_finish` events (backend-reported cost):

| agent | steps | minutes | cost |
|---|---|---|---|
| designer (cycle 1) | 59 | 60 | $3.54 |
| critic | 105 | 29 | $1.50 |
| theorist 1 | 95 | 29 | $1.28 |
| theorist 2 | 82 | 27 | $1.24 |
| theorist 3 | 101 | 33 | $1.72 |
| **total** | **442** | | **$9.28** |

**The pilot's Gemini 3.8 Flash calls are not counted anywhere.** `pilot.py` calls `record_usage`,
but `src.rsa_open` never calls `token_usage.start_usage_log`. The records were kept only in the
job's memory and lost when it ended, and the same holds for the agents' own records. The
transcripts' costs above survive only because opencode writes them. **Before run B, the chain
should write `token_usage.jsonl`, as `src/rsa/loop/run.py` does.**

A side note: every agent's first step was `hostname && echo $SLURM_JOB_ID`, following Sherlock's
own instructions to automated tools (`/etc/agents/AGENTS.md`). It cost one step each.

## 6. The temperature audit (all of E, 50,587 trials; 4 chains × 1000 + 1000; 6 h limit per fit)

`elpd` is free minus fixed (PSIS-LOO, ± SE clustered by display). Positive means a free temperature
predicts E better.

| theory | fixed: s, div | free: s, div | free T | elpd free − fixed |
|---|---|---|---|---|
| evaluative_prominence_l2 | 3128, 0 | 3520, 0 | 1.92 | **+71.6 ± 28.9** (2.5 SE) |
| rsa_l1 | 1550, 0 | 1731, 0 | 1.46 | +49.9 ± 28.5 (1.8 SE) |
| rsa_l1_salience | 2189, 0 | 2678, 0 | 1.85 | +73.3 ± 43.9 (1.7 SE) |
| rsa_l2 | 2757, 0 | 3114, 0 | 1.36 | +26.6 ± 19.4 (1.4 SE) |
| competitor_confusion_speaker | 2458, 0 | 2437, 0 | 1.22 | +12.5 ± 11.6 |
| the other 7 that finished both ways | | | 0.90–1.14 | within ±1 nat, none beyond 2 SE |
| rsa_l1_shared_prior | 2671, 0 | **timed out (6 h)** | | — |
| softmax_surprisal_isolated_costly_l4 | 16452, 0 | **timed out (6 h)** | | — |
| oddity_heuristic_listener | 760, 0 | 2969, 9, not converged | 0.96 | +0.1 ± 0.1 |

What I read from it (the decision is yours and the PI's):
- **The fixed fits are all healthy.** Every one finished, with 0 divergences.
- **Most theories lose nothing by the fix.** The free temperature lands at 0.9–1.15.
- **Five theories want a sharper link.** Free T is 1.2–1.9, and they gain 12–73 nats. Only
  `evaluative_prominence_l2` clears 2 SEs; `rsa_l1` and `rsa_l1_salience` come close. I have not
  checked whether these five lack a sharpness parameter of their own. If they do, the theorists'
  prompt now allows one (f62d19e5), and that could absorb the gain.
- **Free fits stall.** Two time out, and one does not converge. This is the instability that
  motivated the fix.
- **For run B's design fits on all of E,** the slowest fixed fit, `softmax_surprisal_isolated_costly_l4`,
  takes 4.6 h at the audit's 8,000 iterations. A design fit runs 1,500 (2 × 750), so it should take
  roughly 50 min, inside the 2 h limit. That estimate is mine, not measured.

## 7. A gap in the restart steps

Restarting into an existing chain whose corpus changed fails. `write_corpus` refuses to overwrite
`experiment<n>/prior_corpus.json` ("a corpus file is written once"). It does not archive the
attempt as `design.abandoned_<i>`. Run A's restart steps did not cover this, so I moved the stale
directories aside by hand. Run B starts in a fresh root, so it won't meet this. A future restart
after a corpus change will.

## 8. Next

Run B waits for your review. When it is time, it uses a fresh root, `rsa_open_rehearsal`, with
LITERATURE=all. It must not copy this ground truth, which was fitted on pragmods only.

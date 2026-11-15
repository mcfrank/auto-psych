# Stage 1: open chain 1 and fixed chain 1 (2026-11-13 to 11-15)

Written by Claude Opus 5.5 from the run trees, job logs and the gate's output; not reviewed by a person.

**Bottom line.** Both chains ran their three experiments live, in one job each, with no failure, no continuation job
and no person in the loop after the PI's review of the open chain's first paradigm. **The registered gate fails** on a
single item: the open chain's experiment 2 (two 7-point ratings) does not separate the carried incumbent from the
chance theory (1.2 clustered SEs; more than 4 needed). Every other item passes for both chains, and the fixed chain
passes all of them. Under `PREREG_V2.md` §4 ("If either fails"), no further chains run under this registration, a
revision is registered anew, and this pair is reported as a pilot. On that experiment's data **no theory, carried or
new, gets more than 2.3 SEs from chance** (below), so the rating paradigm carried little display-level signal, not
only signal that the carried theory failed to transfer.

Registration https://osf.io/wurh5, tag `rsa-open-prereg-v2` (b32e732b). The gate ran from the Sherlock checkout at
the tag (clean); its `gate.py` sha256 matches the registered one (`0ae77936…`). Aggregates are in
`data/rsa_open/live_v2/` (the chain-1 handoff's "Bring back" list; `design/frozen/` without `lists.json` and the page
preview), with the gate's JSON as `data/rsa_open/live_v2/gate_stage1.json`. As in chain 1's copies,
`participants.json` keeps the campaign's pseudonymous ids (`p###`) of excluded and flagged people; Prolific ids, the
id map, trial-level data and comments stayed on Sherlock.

## The gate (`python -m src.rsa_open.gate`, `PREREG_V2.md` §4)

| item | open chain 1 | fixed chain 1 |
|---|---|---|
| 1 design | pass ×3 (the designer's first cycle each time) | pass ×3 (frozen spec) |
| 2 data: exclusions | 5.5%, **21.5%**, 9.5% (pass; at most 25%) | 11.7%, 15.3%, 11.5% (pass) |
| 2 data: comprehension, catch | 97/99%, 89.5/92.5%, 96/96% | 97.5/94%, 98/92%, 99/94% |
| 2 data: same answer on ≥90% of test trials | 0, 2.0%, 0 | 0, 0.3%, 0 |
| 3 information: first incumbent over chance (clustered SEs; >4) | 7.2, **1.2 FAIL**, 11.9 | 7.4, 9.2, 9.3 |
| 4 loop: trusted at the end, export | 4, 5, 7 | 4, 8, 6 |
| 4b new theories admitted over 1-3 | 52 | 53 |
| 5 time per experiment (h; at most 48) | 9.2, 13.6, 13.9 | 7.7, 8.0, 9.2 |
| reported: chain's running time | 37 h | 25 h |
| reported: agent spend | $114.49 | $126.31 |
| reported: participant payments | $800 | $800 |

### The failing item

Experiment 2's incumbent at its first scoring was experiment 1's winner, `extension_noisy_graded_semantics_l2`:
261.1 nats ahead of chance on the new block, clustered SE 209.5 (50 displays). Of the four carried theories, the
spread was 1,069 nats (2.4 SEs), reported and not judged. Scoring every theory in `newest_block_cv.json` against
chance the gate's way (out-of-fold, `cluster_dse` over displays), not part of the gate:

| experiment | best theory over chance, any | best carried theory | largest single display |
|---|---|---|---|
| open 1 (choice) | 8.0 SE (1,737 nats) | 7.3 SE | 114 nats |
| **open 2 (dual rating)** | **2.3 SE (913 nats)** | 2.2 SE (1,051 nats, the singleton-salience seed) | **259-320 nats** |
| open 3 (production) | 22.7 SE (3,326 nats) | 13.6 SE | 96 nats |
| fixed 1-3 (choice) | 8.0, 9.5, 9.4 SE | 7.4, 9.2, 9.4 SE | 68-72 nats |

On the rating data the theories' advantage over chance comes mostly from one display (about 320 nats of a best total
of about 1,040), so the clustered SE is large for every theory. Whether that is the paradigm (two ratings of
highlighted objects may carry less display signal per person than a choice), the chosen displays, or the link the
theories use for ratings, this run does not settle. **For the revision:** the item
as registered asks the experiment's first incumbent to clear 4 SEs, but here no theory came near that, so a fix to
the incumbent rule alone would not have passed it.

## Timeline (PT)

| | open chain (job 52234617) | fixed chain (job 52234621) |
|---|---|---|
| exp 1 design (test deploy) | 11-13 10:13-12:21 (job 52209295) | 10:13-10:45 (job 52209415) |
| exp 1 collection | published 14:09, done 15:10 | published 14:10, done 15:43 |
| exp 1 loop | 15:10-22:16 (7.1 h) | 15:43-22:52 (7.2 h) |
| exp 2 design | 22:16-01:29 (11-14) | 22:52-23:25 |
| exp 2 collection | 01:29-04:56 | 23:25-03:36 |
| exp 2 loop | 04:56-15:16 (10.3 h) | 03:36-11:04 (7.5 h) |
| exp 3 design | 15:16-17:52 | 11:04-11:52 |
| exp 3 collection | 17:52-20:22 | 11:52-13:57 |
| exp 3 loop | 20:22-07:41 (11-15; 11.3 h) | 13:57-22:24 (8.5 h) |
| job elapsed | 1 d 17 h 36 min | 1 d 8 h 18 min |

Each loop ran 3 rounds of 6 with its fits inside the job (`STEP_FITS=local`: 23 and 22 fit steps over the chains, each
followed by a resume in the same job); no continuation job, no array job, and neither log mentions an out-of-memory
kill. Both logs end `STOP: finished` and both cells have `DONE`; Slurm lists both jobs as
FAILED, exit code 10, as it listed the test deploys' exit 14.

## Designs

| | open 1 | open 2 | open 3 | fixed 1-3 |
|---|---|---|---|---|
| paradigm | `pragmatic_choice_and_rating_7pt` | `pragmatic_dual_rating_7pt` | `pragmatic_speaker_production` | "Which one does Bob mean? (forced choice)" |
| response | choice | two 7-point ratings | production (choose a word) | choice |
| domains | robot, house, snowman | same | same | friend, snowman, sundae, pizza, boat, Christmas tree |
| N; trials (distinct) + catch | 200; 25 (50) + 2 | 200; 25 (50) + 2 | 250; 30 (60) + 2 | 400; 13 (52) + 2 |
| pay, declared minutes | $1.00, 5.5 | $1.00, 5.5 | $0.80, 4.5 | $0.50, 3.0 |
| EIG (bits; per $, floor) | 3.57 (0.0134, 0.0062) | 3.97 (0.0149, 0.0069) | 3.83 (0.0144, 0.0070) | 3.54 / 3.89 / 3.97 |
| pilot (12 simulated): comp, catch, est. min | 1.0, 1.0, 5.5 | 1.0, 1.0, 5.5 | 1.0, 1.0, 4.7 | 1.0, 1.0, 2.7-2.9 |

All three open paradigms were `submitted_by: designer` in its first cycle; only experiment 1 needed review. Open
experiment 3's minutes were calibrated on "7 experiment(s) of other response types ... (none of production yet)";
people took 6.3 min at the median against 4.5 declared.

## Collection

| | open 1 | open 2 | open 3 | fixed 1 | fixed 2 | fixed 3 |
|---|---|---|---|---|---|---|
| recruited, kept | 200, 190 | 200, 157 | 250, 228 | 400, 354 | 400, 338 | 400, 354 |
| excluded: catch, comprehension | 5, 6 | 25, 21 | 16, 9 | 41, 10 | 58, 8 | 45, 4 |
| minutes, median (p10, p90) | 4.7 (3.0, 9.2) | 6.6 (3.8, 13.1) | 6.3 (3.5, 11.4) | 2.6 (1.5, 5.3) | 3.1 (1.8, 7.0) | 2.8 (1.6, 5.9) |
| pay at the median | $12.70/h | **$9.06/h** | **$7.58/h** | $11.50/h | **$9.55/h** | $10.68/h |
| enjoyment (1-5), mean | 3.87 | 4.32 | 3.86 | 4.08 | 4.23 | 4.09 |
| returning (another chain's study) | 0 | **138 (69%)** | 71 (28%) | 92 (23%) | 8 (2%) | 13 (3%) |

(Exclusion reasons overlap, so the two counts can exceed the total.) "Returning" is the registered behaviour: a
chain's blocklist covers only its own earlier studies, and people from the other chain are kept and counted for the
sensitivity analysis (`live.py`). Prolific studies: open 6af78aa1…, 6af82a44…, 6af91082…; fixed 6af78ac3…,
6af80d4d…, 6af8bc36….

## The loops

| | admitted | rejected | pruned | too slow, not converged | carried | incumbent and `fit_best` (parameters) | runner-up |
|---|---|---|---|---|---|---|---|
| open 1 | 18 | 3 (2 contract, 1 not novel) | 20 | 0, 0 | 4 | `extension_noisy_graded_semantics_l2` (`alpha` LogNormal(0,1), `w_salience` Normal(0,1)) | 21.9 ± 18.6 |
| open 2 | 16 | 5 (2 no file, 2 not novel, 1 contract) | 15 | 0, 0 | 5 | `l1_graded_semantics_l2` (same two) | 1.2 ± 1.7 |
| open 3 | 18 | 0 | 16 | 0, 0 | 7 | `complexity_gated_retrieval_decision_l2` (same two) | 1.0 ± 1.8 |
| fixed 1 | 18 | 0 | 20 | 0, 0 | 4 | `prototype_tier_confusion_l2` (`alpha`, `w_confusion` LogNormal(0,0.5)) | 0.3 ± 3.8 |
| fixed 2 | 17 | 1 (no file) | 13 | 0, 0 | 8 | `contrastive_anchor_decision_l2` (`alpha`, `w_dec` LogNormal(0,0.5)) | 0.6 ± 3.7 |
| fixed 3 | 18 | 1 (not novel) | 20 | 0, 0 | 6 | `sole_exclusion_shadowing_heuristic` (`w_crowd` LogNormal(0,0.5)) | 5.8 ± 6.5 |

Runner-up: nats behind the incumbent on the experiment's block ± clustered SE. The incumbent changed at every
experiment boundary in both chains, and within every experiment; every theory carried at the end is trusted. Round
by round (incumbent after the round; critique statistics significant of 8, after FDR):

- open 1: `competitor_confusion_speaker` (seed) → `competitor_confusion_shared_prior` 6 (6) → `noisy_graded_semantics_l2` 6 (6) → `extension_noisy_graded_semantics_l2` 7 (7)
- open 2: carried → `l1_graded_semantics_l2` 8 (8), kept through rounds 2-3 (5 (5), 4 (3)); experiment 1's winner ended 780 ± 326 behind
- open 3: carried → `extension_cost_speaker_l2` 8 (8) → `graded_lexical_uncertainty_l2` 5 (3) → `complexity_gated_retrieval_decision_l2` 1 (0)
- fixed 1: seed → `competitor_confusion_l2` 7 (7) → `tier_competitor_confusion_l2` 7 (7) → `prototype_tier_confusion_l2` 6 (5)
- fixed 2: carried → `prototype_anchored_tier_qud_l2` 5 (2) → `prototype_span_false_access_l2` 6 (6) → `contrastive_anchor_decision_l2` 4 (0)
- fixed 3: carried → `minimal_sufficiency_shadowing_l2` 5 (5) → `soft_lexical_truth_l2` 3 (0) → `sole_exclusion_shadowing_heuristic` 4 (0)

The open chain's three incumbents have the same two parameters (`alpha`, `w_salience`); experiment 2's refined
experiment 1's. Admission limit 30 min
throughout; nothing hit it.

## Spend

| | priced calls | estimate for unpriced calls | total |
|---|---|---|---|
| open | $45.45 (2,084 calls: candidates $41.07, critique $1.42, designer $2.96) | $69.04 (66 records) | $114.49 |
| fixed | $13.82 (717 calls: candidates $11.58, critique $2.23) | $112.49 (61 records) | $126.31 |

The pilot's participant model (`gemini-3.8-flash`, 2,001 open and 648 fixed calls) has no priced record, so it is
neither priced nor estimated. Participants: $800 a chain (the gate's figure), about $1,600 in all. No usage-limit
waits.

## For the driver

1. **The gate failed; the revision is the next step** (`PREREG_V2.md` §4). The information item's evidence is above:
   on the dual-rating data nothing reached 4 SEs, and one display carried most of the signal.
2. **The handoff's `python -m ...` commands** (review, monitor, status, gate) fail on the login node, whose `python`
   is 2.7. They need the agent tree's venv,
   `$SCRATCH/auto-psych/agent_trees/$(cat $WORK_ROOT/cells/<arm>_chain_0/agent_tree_id)/venv/bin/python`, from the
   checkout. `src.rsa_open.status` also needs `--work-root`.
3. **Pay per hour fell below the listing** in open 2-3 and fixed 2 ($7.58-9.55/h). Open 3's time was calibrated with
   no production experiment behind it and came in 40% over.
4. **Cross-chain overlap is large**: 69% of open 2's sample had done another chain's
   study, and 23% of fixed 1's (the studies opened within a minute of each other). Registered and counted, but the
   sensitivity analysis will lean on it.
5. **Open 2's exclusions (21.5%)** were close to the 25% bound.
6. **Log warning** `os.fork() was called ... JAX is multithreaded, so this will likely lead to a deadlock`, 34 and 33
   times (at step resumes) in the two logs. No deadlock occurred.
7. **Slurm reports a finished chain as FAILED** (exit 10), like a test deploy (exit 14); `src.rsa_open.status` and
   the cell's `DONE` are the record.

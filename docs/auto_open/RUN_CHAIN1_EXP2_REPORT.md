# Open chain 1 (`chain_0`), experiment 2 (2026-10-31 to 11-03)

Written by Claude Opus 5.5 from the run tree and job logs; not reviewed by a person.

**Bottom line.** The study went cleanly (151 kept of 163, filled in 79 min). The loop took 72.3 h, of which about 37 h went
to two out-of-memory failures, and it finished on tag `rsa-open-prereg-a2`. The leader changed in all 5 rounds, as in
experiment 1. The final leader came from an exploratory slot in the last round and leads by only about 1.7 SE, so
experiment 2 did not settle on a theory. Experiment 3's design started by itself at 17:23 PT on 11-03.

Aggregates are in `data/rsa_open/live/open_chain_0/` (the handoff's "Bring back" list, except as noted at the end).
Experiment 1's report is `RUN_CHAIN1_EXP1_REPORT.md`; the failures and time are in `RUN_CHAIN1_STATUS_2026-11-02.md`.

## Design

| | |
|---|---|
| design stage | job 50237409, 3.2 h (13:34-16:48 PT, 10-31); no review required after experiment 1 |
| paradigm | `pragmatic_choice_and_probed_rating` (sha256 `e1308d30…`), `submitted_by: designer` |
| response | forced choice of a referent, then a 7-point rating of a probed candidate |
| materials | SVG, 3 domains |
| trials | 30 test + 2 catch (+ 1 practice) |
| N, pay, minutes | 163, $1.22, 7.1 min declared; about $265 with fees |
| EIG | 4.02 bits, 0.0152 bits per dollar (floor 0.0070) |
| pilot (`gemini-3.8-flash`, 12 simulated people) | comprehension 100%, catch 100%, readable 100%, 7.2 min estimated |

## Collection

| | |
|---|---|
| Prolific study | 6ae67e7a786c10561f49ae66, published 16:49 PT, filled 18:07 PT (79 min) |
| blocklist | experiment 1's study 6ae41a81d525be2a1c9901d6 (checked on the live study) |
| responses, kept | 163 (163 recruited), 151 |
| excluded | 12 (7.4%): missed a catch trial 6, comprehension on the second reading 6 |
| comprehension, catch | 96.3%, 98.2% |
| same answer on nearly every trial, fast, returning | 0, 0, 0 |
| minutes, median (p10, p90) | 6.1 (4.0, 11.2) |
| pay at the median | $11.91/h |
| enjoyment (1-5) | mean 3.71, n 163: 7 / 17 / 41 / 50 / 48 |
| Prolific balance afterward | $2,590.59 |

## The loop

72.3 h elapsed, 18:07 PDT on 10-31 to 17:23 PST on 11-03 (daylight saving ended on 11-01), against a registered estimate of about 20 h. Admission limit
125 min (2 × the median starting fit). Five rounds of six slots; **the leader changed in every round**, so the stale-round
stop never applied.

| after | leader | came from | admitted, rejected | critique: significant of 8 (after FDR) |
|---|---|---|---|---|
| seeds | minimal_exemplar_diagnostic_mixture_l2 (carried) | experiment 1 | 4 carried | – |
| round 1 | prototype_figure_individuated_mixture_l2 | refine chosen, retry | 5, 1 (no file) | 6 (6) |
| round 2 | polar_figure_ground_topic_tom | explore | 6, 0 | 5 (4) |
| round 3 | r3_c4_2 | refine incumbent, repair | 6, 1 (not novel) | 3 (2) |
| round 4 | bipolar_boundary_spillover_topic_tom_2 | refine incumbent, repair | 6, 1 (not novel) | 4 (3) |
| round 5 | **attentional_buffer_anchor_sampling** | explore, retry | 6, 1 (not novel) | 4 (2) |

Ledger: 58 new lines, for 122 in all. 29 admitted; 4 rejected, of them 3 not novel (each repaired and then admitted) and 1
slot whose agent wrote no `candidate.py` even on retry; 25 removed at the end (15 by margin on exp2, 2 × clustered SE; 9
ineligible, more than 4 SE behind the best on an earlier block; 1 retired by the cap of 8). **None was too slow to fit**
(10 of 37 attempts in experiment 1, at a limit of 73 min).

**Exported live set** (`export.json`), grouped-CV ELPD behind the best on each block, in nats (SE). Selection uses exp2
only; E and exp1 are the guard.

| model | source | exp2 (selection) | E | exp1 |
|---|---|---|---|---|
| attentional_buffer_anchor_sampling (best) | round 5, explore | 0 | 11.5 (5.8) | 103.8 (59.5) |
| singleton_anchored_polar_boundary_tom | round 5, refines round 4's leader | 8.2 (4.9) | 9.1 (7.1) | 13.7 (19.3) |
| bipolar_prototype_tier_contrast_tom | round 5, refines round 4's leader | 8.4 (4.7) | 4.8 (7.1) | 10.8 (19.1) |
| topic_worthiness_boundary_tom | round 5, explore | 14.7 (17.6) | 0, kept as the best on E | 317.7 (147.7) |
| prototype_boundary_listener_mixture | round 4, explore | 43.1 (37.3) | 33.6 (16.2) | 323.0 (116.2) |
| bipolar_clone_boundary_spill_tom | round 5, refine chosen | 43.3 (37.8) | 15.2 (5.5) | 224.1 (100.1) |
| lexical_blocking_fallback_cost_tom_2 | round 5, explore, repair | 45.3 (38.3) | 9.6 (6.3) | 230.4 (101.9) |
| minimal_exemplar_diagnostic_mixture_l2 | carried from experiment 1 | 161.9 (24.8), ineligible | 259.3 (55.6) | 0, kept as the best on exp1 |

All of experiment 1's other exports are gone: `speaker_alternative_necessity_tom` and `heterogeneous_listener_type_mixture`
were ineligible on E, and the last seed, `rsa_l2_singleton_feat_color_valence_l0`, was ineligible on exp1 (197.9 ± 48.2).

## Time and spend

| | hours |
|---|---:|
| loop attempt 1, then stopped (out of memory at 96 GB; restarted from scratch) | 20.6 lost |
| loop attempt 2, seed fits and rounds 1-2 kept | 12.1 |
| round 3, first try, then stopped (out of memory at 180 GB; fix written and tested) | 16.3 lost |
| round 3 on tag a1 | 6.7 |
| round 4 on tag a2, with about 35 min to resume | 7.5 |
| round 5 and the end of the experiment | 8.9 |
| **total** | **72.3** (35.4 without the failures) |

Both failures were infrastructure, so under §4a they resumed rather than restarting the chain. Addenda B and C (tags a1 and
a2) are posted on OSF; no setting, seed, cache key, gate or score changed. Running time so far, counted as item 5 counts
it: **116.6 h** through this loop's end, now reported rather than judged under Addendum C.

Agent spend so far, recomputed from `token_usage.jsonl`: **$281.64 priced, plus about $27 estimated for unpriced pilot
calls, about $309 of the $495 limit.** Experiment 2's loop took about $169 of that (candidates and critiques), about $64
of it in the two failed attempts. Experiment 3 will add a pilot's unpriced calls and a loop, so my rough projection stays
about $435. The gate script will give the official figure.

Memory of the job on the lab node: at most 151.6 GB of 184 (`sstat` MaxRSS). The loop's own process peaked at 23.6 GB
(`VmHWM`) with 33 live theories, below the 31 GB I had extrapolated from 21 theories.

## For the driver

1. **The theories are not saturating.** The leader has changed in 10 of 10 rounds across two experiments. Six of the 8
   exports, including the top four by exp2, were admitted in the last round. A later round might well have changed it again.
2. **The leader is not resolved, and the rule favors it over two more balanced theories.** It is 8.2-8.4 nats (about 1.7
   SE) ahead of two refinements of the previous leader. Those two are within 2 SE of the best on every block, while the
   leader is 104 nats behind on exp1 (1.7 SE, so eligible, with a very large SE). Selection uses the newest block only, by
   design. Experiment 3's prospective score is the real test: every one of these was selected on exp2.
3. **The conflict in experiment 1's report has eased.** There, discovered theories were 171-185 nats behind the seed on E.
   Now the best on E is itself discovered, and the top three are within about 12 nats of it. Exp1 is the block still in
   tension: its best theory is 162 nats behind on exp2 and ineligible on E, and it survives only as the guard's reference.
4. **Several pruning decisions were close.** The previous leader was pruned at 11.1 nats against a threshold of 10.6 (2 ×
   5.3). `bipolar_prototype_tier_contrast_tom` was kept at 8.4 against 9.4, `singleton_anchored_polar_boundary_tom` at 8.2
   against 9.8, and `polar_grounding_boundary_spill_tom` was pruned at 14.3 against 11.8. The seed-noise test now running
   (`DISTRIBUTED_DESIGN_REVIEW_V3.md` §10) measures fold noise at round 4's leaders, which bears on calls this close. It
   cannot test these exact calls, because it doesn't include round 5's theories.
5. **Nothing was too slow this time.** That is confounded: the limit was 125 min rather than 73, because it is 2 × the
   median starting fit and the slow seeds set it.
6. **Pay ran over target, the reverse of experiment 1:** 6.1 min median against 7.1 declared, $11.91/h. Experiment 1 was
   under, at $9.27/h. The pilot's calibration here came from other response types.
7. **`token_usage_summary.json` is stale.** It was last written at 08:36 on 11-02 and shows $216.08, so the loop's later
   agent calls aren't in it. I brought it back as listed and used `token_usage.jsonl` for the figures above. The driver may
   want the summary rewritten at each stage.

## Decisions I made, and what to check

- I counted the time lost using the status note's breakdown (attempt 1 plus its stop, round 3's first try plus its stop).
  Round 3's first-try hours come from that note, not from a new measurement.
- "$64 in the failed attempts" takes all of attempt 1 ($43.58) and a third of attempt 2 ($60.11), the share of its time
  spent on the lost round; that split is an estimate.
- As in experiment 1, `newest_block_cv.json` is per-trial: log-likelihoods for the 4,530 kept trials under each model, by
  the campaign's person and trial numbers, with no responses and no Prolific ids. The handoff lists it, so it is brought
  back; say if it should stay on Sherlock.

Not brought back: `lists.json` (the per-person stimulus lists), `page_preview.html` (a page file), and the pilot's simulated
rows and transcripts (only `pilot/pilot.json`).

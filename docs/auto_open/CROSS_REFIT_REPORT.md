# Every stage-1 theory on every stage-1 paradigm: the cross refit (2026-11-16)

Written by Claude Opus 5.5 from the Sherlock run of `HANDOFF_cross_refit.md`; not reviewed by a person. Descriptive
only: stage 1 is a pilot, and this is a dry run of the bakeoff's fit A (`PREREG_V2.md` §6) on its data.

**Bottom line.** Each arm's final theory is the best of the 13 on its own arm's paradigms (except the dual rating, below), and the other arm's final
is more than 2 clustered SEs behind it on every live block: 4-5 SE on the fixed chain's three choice experiments,
2.3 SE on the open chain's choice-and-rating experiment, 6 SE on its production experiment. On the literature (E)
the two finals are level (1.4 SE apart; the fixed arm's second incumbent leads). The exception is the open chain's
dual-rating experiment (`open_exp2`), where no theory does much better than chance (at most 2.4 SE), the open arm's
own final loses to plain RSA by 456 nats (2.5 SE), and the best theory is the fixed arm's *first* incumbent. The
fixed arm's final, the winner on all three fixed blocks, **did not converge** and took five times longer than any other
theory to fit; its R-hat and divergences are being recovered by a diagnostic re-fit (below).

## The run

- Code `auto-psych-open` at 5066bcde (fits) and d11953f4 (report; `cross_refit.py` is unchanged between them).
- Data: the two chains' `experiment3/corpus.json`, copied; the script checked both hold the same literature block E.
  Seven blocks: E, `open_exp1` (choice and rating), `open_exp2` (two 7-point ratings), `open_exp3` (production),
  `fixed_exp1`-`fixed_exp3` (forced choice).
- 13 theories (`chance`, six bar references, each chain's incumbent after each experiment, prefixed `open<k>` /
  `fixed<k>`), each fitted jointly to all seven blocks with grouped 5-fold CV inside every block (whole units per
  fold), 4 chains × 1,000 + 1,000, dense mass, target accept 0.9, seed 0: the loop's settings.
- A 13-task Slurm array on `mcfrank`, 2 CPUs and 15.6 GB per task. The per-trial `.npz` scores stay on Sherlock
  (`$SCRATCH/auto-psych/refits_1115/cross_out/`).
- Aggregates here: `data/rsa_open/live_v2/cross_refit/cross_refit.{md,json}` (the script's table) and
  `cross_refit_pairwise.json` (each block's best theory and both finals against it, with clustered SEs; computed
  from the same `.npz` files).

## 1. The table

Each cell: out-of-fold elpd over chance (its ratio to the unit-clustered SE); then over plain RSA (`rsa_l1`) and
its ratio.

| theory | E | open_exp1 | open_exp2 | open_exp3 | fixed_exp1 | fixed_exp2 | fixed_exp3 |
|---|---|---|---|---|---|---|---|
| literal_listener | 28627 (15.6); -1372 (-6.2) | 1223 (6.2); -293 (-4.6) | 962 (2.2); -17 (-0.5) | 3236 (22.7); +294 (+3.9) | 684 (4.7); -249 (-2.9) | 523 (4.4); -364 (-3.2) | 483 (4.5); -427 (-4.3) |
| rsa_l1 | 29999 (16.1) | 1516 (6.6) | 978 (2.3) | 2942 (17.3) | 933 (5.0) | 887 (5.4) | 910 (6.3) |
| rsa_l2 | 30040 (16.2); +41 (+4.9) | 1527 (6.7); +11 (+2.6) | 981 (2.3); +2 (+1.4) | 2839 (15.9); -103 (-5.2) | 951 (5.1); +18 (+2.0) | 859 (5.4); -28 (-1.3) | 905 (6.3); -5 (-0.5) |
| rsa_l2_singleton_feat_color_valence_l0 | 30547 (16.5); +548 (+5.2) | 1561 (6.8); +45 (+1.3) | 1016 (2.4); +38 (+1.5) | 2558 (13.0); -384 (-3.8) | 1174 (6.7); +241 (+4.5) | 1271 (8.3); +384 (+5.7) | 1200 (8.7); +290 (+3.8) |
| competitor_confusion_speaker | 30313 (16.3); +314 (+5.0) | 1601 (7.2); +85 (+4.3) | 965 (2.2); -13 (-1.2) | 2870 (17.4); -71 (-1.5) | 1035 (5.9); +102 (+2.7) | 1111 (7.2); +224 (+3.5) | 1141 (7.9); +231 (+3.1) |
| softmax_surprisal_isolated_costly_l4 | 30285 (16.4); +285 (+3.9) | 1591 (7.2); +75 (+3.0) | 975 (2.2); -3 (-0.2) | 2882 (16.5); -60 (-1.5) | 1146 (6.8); +213 (+4.7) | 1263 (8.4); +376 (+5.6) | 1204 (8.6); +294 (+3.7) |
| open1_extension_noisy_graded_semantics_l2 | 30632 (16.5); +633 (+7.3) | 1781 (7.6); +265 (+6.8) | 414 (1.6); -564 (-2.5) | 2403 (10.5); -538 (-4.8) | 1177 (6.6); +244 (+5.7) | 1196 (7.7); +309 (+6.4) | 1153 (8.3); +243 (+3.4) |
| open2_l1_graded_semantics_l2 | 30408 (16.4); +409 (+4.9) | 1650 (7.7); +134 (+4.3) | 1001 (2.3); +22 (+1.6) | 2414 (11.4); -528 (-5.0) | 1169 (6.7); +236 (+5.2) | 1212 (8.0); +326 (+6.0) | 1164 (8.3); +254 (+3.5) |
| open3_complexity_gated_retrieval_decision_l2 | 30648 (16.5); +648 (+6.1) | 1798 (7.8); +282 (+6.9) | 522 (1.8); -456 (-2.5) | 3340 (22.7); +398 (+5.2) | 1208 (7.0); +275 (+5.1) | 1252 (8.3); +365 (+6.5) | 1156 (8.2); +246 (+3.4) |
| fixed1_prototype_tier_confusion_l2 | 30511 (16.5); +511 (+4.9) | 1707 (8.1); +191 (+4.9) | 1018 (2.4); +40 (+2.3) | 2382 (12.0); -560 (-5.1) | 1302 (7.8); +369 (+6.3) | 1321 (8.8); +434 (+7.0) | 1263 (9.1); +353 (+4.9) |
| fixed2_contrastive_anchor_decision_l2 | 30749 (16.6); +750 (+5.8) | 1718 (8.1); +202 (+5.2) | 1006 (2.3); +28 (+1.3) | 2347 (11.7); -595 (-5.0) | 1311 (7.9); +378 (+6.2) | 1376 (9.2); +489 (+7.3) | 1287 (9.0); +377 (+5.0) |
| fixed3_sole_exclusion_shadowing_heuristic (not converged) | 30736 (16.7); +737 (+5.0) | 1696 (7.8); +180 (+4.3) | 426 (1.7); -552 (-2.1) | 2465 (12.6); -477 (-4.0) | 1324 (8.0); +391 (+6.2) | 1402 (9.3); +516 (+7.2) | 1404 (9.3); +494 (+6.1) |

**Not converged: `fixed3_sole_exclusion_shadowing_heuristic`**, the fixed arm's final, the only one of the 13. The fit
record keeps only the flag (the loop's gate: more than 0.1% divergent transitions, R-hat above 1.05 or bulk ESS
below 100 in any of the five fold fits), not the numbers. A diagnostic re-fit with the same data, folds, settings and
seed, recording each fold's divergences, R-hat and ESS, is running on Sherlock (job 52525490, about 11 hours); its
numbers will be added here. Until then, read its column as the score of an unconverged fit: its wins on the fixed
blocks may be real or may be partly a sampling artifact.

## 2. Each block's best theory, and the other arm's final

Difference from the block's best theory, out-of-fold elpd (unit-clustered SE; ratio).

| block | units | best | open arm's final (`open3`) | fixed arm's final (`fixed3`) | other arm's final within 2 SE? |
|---|---|---|---|---|---|
| E | 422 | `fixed2_contrastive_anchor_decision_l2` | −102 (72; 1.4) | −13 (48; 0.3) | yes (both finals within 2 SE) |
| open_exp1 | 50 | `open3_complexity_gated_retrieval_decision_l2` | best | −102 (45; 2.3) | no |
| open_exp2 | 50 | `fixed1_prototype_tier_confusion_l2` | −496 (183; 2.7) | −592 (265; 2.2) | neither final is within 2 SE |
| open_exp3 | 60 | `open3_complexity_gated_retrieval_decision_l2` | best | −875 (144; 6.1) | no |
| fixed_exp1 | 52 | `fixed3_sole_exclusion_shadowing_heuristic` | −116 (27; 4.3) | best | no |
| fixed_exp2 | 52 | `fixed3_sole_exclusion_shadowing_heuristic` | −150 (35; 4.3) | best | no |
| fixed_exp3 | 52 | `fixed3_sole_exclusion_shadowing_heuristic` | −248 (46; 5.4) | best | no |

Reading it:

- **The answer to the handoff's first two questions is no, in both directions.** The fixed arm's theories, built on
  one forced-choice paradigm, do not carry to production: every one of them is 477-595 nats behind plain RSA on
  `open_exp3`, where only the open arm's final (+398, 5.2 SE) and the literal listener (+294, 3.9 SE) beat plain RSA. The open arm's
  final does not carry to the fixed paradigm either: it is 4-5 SE behind `fixed3` on each fixed block, though it still
  beats plain RSA there by 246-365 nats.
- **The dual-rating block (`open_exp2`) discriminates poorly.** Every theory is at most 2.4 SE over chance, and the
  ratios against plain RSA are mostly within ±2.5. Both finals, and the open arm's first incumbent, lose to plain RSA
  there by 456-564 nats with SEs of 180-270: something in their rating link costs them on these 50 displays. This is
  the same experiment whose information item failed the stage-1 gate (`STAGE1_REPORT.md`).
- **On the literature both arms' later incumbents gain 409-750 nats over plain RSA** (5-6 SE), and the best three
  (`fixed2`, `fixed3`, `open3`) are within 1.4 SE of each other.
- Each block has 50-60 units (displays); the SEs are clustered on them, so the live-block comparisons rest on few
  independent units.

## 3. Time and memory per theory

From `sacct` (elapsed) and the fit records; peak memory is the batch step's MaxRSS. Each task ran five NUTS fits
(one per fold, all seven blocks jointly) on 2 CPUs.

| theory | elapsed | peak memory (GiB) |
|---|---|---|
| chance | 1:18 | 6.2 |
| literal_listener | 0:49 | 6.3 |
| rsa_l1 | 1:02 | 6.6 |
| rsa_l2 | 1:05 | 6.8 |
| rsa_l2_singleton_feat_color_valence_l0 | 1:50 | 6.8 |
| competitor_confusion_speaker | 1:03 | 6.5 |
| softmax_surprisal_isolated_costly_l4 | 2:19 | 7.4 |
| open1_extension_noisy_graded_semantics_l2 | 2:19 | 6.8 |
| open2_l1_graded_semantics_l2 | 1:15 | 6.8 |
| open3_complexity_gated_retrieval_decision_l2 | 2:00 | 6.6 |
| fixed1_prototype_tier_confusion_l2 | 1:15 | 6.8 |
| fixed2_contrastive_anchor_decision_l2 | 1:30 | 7.1 |
| fixed3_sole_exclusion_shadowing_heuristic | **10:53** | 6.7 |

Memory is flat (6-7.4 GiB, dominated by the corpus and JAX); time ranges from 50 minutes to 2.3 hours except for
`fixed3`, whose 10.9 hours go with its failure to converge (long trajectories in a difficult posterior).

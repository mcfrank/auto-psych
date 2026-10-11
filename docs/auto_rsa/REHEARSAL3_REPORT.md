# Rehearsal 3: report (the critique step, with real agents)

From the local Sherlock session, 2026-10-10, for the driver session and the PI.

| | |
|---|---|
| handoff | `HANDOFF_rehearsal3.md` |
| job | 47283774, COMPLETED, 5 h 18 min, no crash |
| code | c3e0c2fc |
| work root | `$SCRATCH/auto-psych/rsa_rehearsal3` (cell `rehearsal_guarded`, agent tree `73191fc0ed7f2937`) |
| ground truth | rehearsal 2's, copied: `crowd_discrim_confusion_chromatic_l2` |
| run | chain 0; 1 experiment of 200 simulated people × 10 of 40 displays; 2 rounds × 6 slots; Gemini 3.8 Flash agents, no network |

Outputs are in `data/rsa/rehearsal3/`, brought back as for rehearsal 2. The critique dirs
are included without `agent.jsonl` and `scratch/`. No trial-level CSVs, fit caches or agent
logs were brought back. The scan for keys, tokens and Prolific ids is clean.

**Summary:**
- **The critique works mechanically.** Both rounds were `critiqued` on the first attempt,
  with 8 of 8 statistics evaluated and none set aside. Every candidate's brief carried the
  critique, and most hypotheses cite one of its discrepancies.
- **Three things need the driver's decision:**
  1. **What the critic looks at.** It scores all 51,530 trials, of which the live trials
     are 4.4%. Guarded selection ranks on the live trials only. Three of the critic's nine
     distinct statistics come entirely from one literature dataset (§3).
  2. **The p values carry no information at this sample size.** 15 of 16 sit at the
     replicate floor (§2), so "8 of 8 significant" is guaranteed rather than informative.
  3. **It costs more time than planned.** The critique took 17 and 31 min, not "a few
     minutes": 15% of wall time and 17% of spend (§5).

## 1. Did each round have a critique?

| round (history step) | `critique.status` | incumbent critiqued | attempts | statistics / evaluated / p ≤ 0.05 / survive FDR |
|---|---|---|---|---|
| 1 (step 1) | `critiqued` | `rsa_l2_singleton_feat_color_valence_l0` (seed) | 1 | 8 / 8 / 8 / 8 |
| 2 (step 2) | `critiqued` | `r1_c4` (round 1's winner) | 1 | 8 / 8 / 8 / 8 |

- No `critique_retry_1/`.
- **No `broken_statistics/` in either round.** Nothing was set aside as too slow, raising
  or non-finite, and every `error` field in `ppc_results.json` is null.
- The seed step (step 0) and the end-of-experiment step (step 3) have `critique: null`, as
  expected.

## 2. What the critic wrote

Each round had 8 statistics and 1,000 posterior-predictive replicates. Round 2 kept 7 of
round 1's statistics, under the same names and with identical code. It replaced
`four_object_maximal_competitor_rate` with `speakers_l2_target_choice_rate`. The observed
values are therefore the same in both rounds (the data did not change); only the incumbent's
null moved.

The tables are sorted by |z|. The p values are two-sided. `floor` marks a p at the
replicate minimum, 2/1001 = 0.002.

**Round 1** (incumbent `rsa_l2_singleton_feat_color_valence_l0`):

| statistic (critic's description, shortened) | observed | model mean (sd) | z | p | q |
|---|---|---|---|---|---|
| `ambiguous_word_foil_choice_rate`: foil chosen when the word fits ≥ 2 objects | 0.0274 | 0.0138 (0.0009) | +14.9 | 0.002 floor | 0.002 |
| `unambiguous_word_accuracy`: the one matching object chosen | 0.989 | 0.973 (0.0014) | +11.8 | 0.002 floor | 0.002 |
| `scalar_intermediate_competitor_rate`: intermediate competitor in ≥ 3-match feature hierarchies | 0.107 | 0.272 (0.019) | −8.5 | 0.002 floor | 0.002 |
| `standard_scalar_logical_choice_rate`: logical competitor on 3-object scalar displays | 0.185 | 0.240 (0.0095) | −5.8 | 0.002 floor | 0.002 |
| `twin_uniform_singleton_choice_rate`: singleton chosen in E9 "twins" when the word fits all | 0.383 | 0.694 (0.058) | −5.3 | 0.002 floor | 0.002 |
| `size_prior_max_feature_rate`: max-feature object on mumble trials, "size" experiment | 0.580 | 0.441 (0.027) | +5.1 | 0.002 floor | 0.002 |
| `four_object_maximal_competitor_rate`: 4-feature competitor vs 1-feature target, 4-object displays | 0.251 | 0.145 (0.021) | +4.9 | 0.002 floor | 0.002 |
| `two_object_pragmatic_target_rate`: fewer-feature target when both of 2 objects match | 0.677 | 0.771 (0.028) | −3.4 | 0.002 floor | 0.002 |

**Round 2** (incumbent `r1_c4`): the seven shared statistics, plus one new one.
- `ambiguous_word_foil_choice_rate`: z +15.7; null 0.0138.
- `unambiguous_word_accuracy`: z +11.3.
- `size_prior_max_feature_rate`: z +5.4.
- `scalar_intermediate_competitor_rate`: z −5.2; null 0.209, down from 0.272.
- **`speakers_l2_target_choice_rate` (new):** "L2 target chosen in the speakers experiment".
  Observed 0.455, null 0.587, z −5.2.
- `twin_uniform_singleton_choice_rate`: z −4.5.
- `standard_scalar_logical_choice_rate`: z −4.3.
- `two_object_pragmatic_target_rate`: z −2.7, p 0.006. This is the only p above the floor.

**Reading:**
- The statistics are substantive. They target the core pragmatic phenomena: ambiguity, the
  scalar implicature's logical and intermediate competitors, the mumble prior, and
  singletons.
- **The top two discrepancies did not move when the incumbent changed.** The foil rate's
  null stayed at 0.0138 and the accuracy's null at 0.973. The new incumbent improved only
  the scalar-intermediate statistic.
- **The p values carry no information.** With 51k trials, a fixed model's misfit on any
  pooled rate is "significant": 15 of the 16 p values sit at the floor, and BH's q changes
  nothing. Only z (or the raw gap) ranks the discrepancies. The brief tells candidates to
  "prioritise discrepancies that survive the FDR", which is every discrepancy here.

## 3. Where the critic's statistics come from (a check I added)

The critic receives every included trial:
- **51,530 rows**: mayn_demberg_2026 16,764; mayn_demberg_2023 15,048; mayn_demberg_2022
  7,590; pragmods 5,390; sikos_2021 4,482; **auto_psych 2,256 (4.4%)**.
- `CRITIQUE_CONTEXT.md` names the sources, and the frame carries `source` and `experiment`.

**The method:** I re-ran each of the critic's statistics on each source alone, using the
same frame as `critique_frame` (`data/rsa/rehearsal3/stat_by_source.py` on a compute node, observed values only).

| statistic | all | auto_psych | literature | where it comes from |
|---|---|---|---|---|
| `ambiguous_word_foil_choice_rate` | 0.027 | 0.018 | 0.028 | mixed (all 5 sources) |
| `unambiguous_word_accuracy` | 0.989 | 1.000 | 0.989 | mixed |
| `scalar_intermediate_competitor_rate` | 0.107 | 0.162 | 0.085 | auto_psych + pragmods |
| `standard_scalar_logical_choice_rate` | 0.185 | 0.224 | 0.180 | auto_psych + pragmods |
| `two_object_pragmatic_target_rate` | 0.677 | 0.755 | 0.624 | auto_psych + pragmods |
| `four_object_maximal_competitor_rate` (r1 only) | 0.251 | 0.251 | — | **auto_psych only** |
| `twin_uniform_singleton_choice_rate` | 0.383 | — | 0.383 | **pragmods only** (`experiment == "E9_twins"`) |
| `size_prior_max_feature_rate` | 0.580 | — | 0.580 | **pragmods only** ("size") |
| `speakers_l2_target_choice_rate` (r2 only) | 0.455 | — | 0.455 | **pragmods only** ("speakers") |

("—" means the statistic has no rows in that source; the code returns 0.0 there.)

**What this shows:**
- **Three of the nine distinct statistics are about one literature dataset's named
  experiments.** Those experiments' manipulations are outside the scope the live designs can
  show.
- **The only statistic wholly on live trials** (`four_object…`, z +4.9) was dropped in
  round 2.
- **The mixed statistics are dominated by the literature by row count.** On some of them
  the live trials sit on the other side of the pooled value: on
  `scalar_intermediate_competitor_rate` the live rate is 0.162, the literature 0.085. So a
  pooled discrepancy need not hold on the live trials.
- I did not compute the incumbent's null per source; that needs the replicates.

**The mismatch in action, round 2:**
- Three of the six slots went after the top discrepancy, the foil rate on ambiguous words.
  All three proposed a trembling-hand speaker.
- Two were rejected as not novel against the third, `trembling_hand_confusion_rsa_l2`
  (RMSE 0.0008 and 0.0007 against the 0.002 threshold).
- `trembling_hand_confusion_rsa_l2` was then the **best model on full-data PSIS-LOO**
  (rank 0, 10.7 nats ahead of the eventual winner). But it was **9.0 ± 4.0 nats behind on
  the live trials' grouped CV**, so guarded selection passed it over and it was pruned at
  the end.
- Guarded selection did what it should. But the critique spent half a round on a mechanism
  aimed at the literature data.

**For the driver:**
- Should the critic see only live trials, or be told to report statistics by `source`?
- Should statistics tied to literature-only experiments (`experiment == "E9_twins"`, etc.)
  be discouraged?
- Should the brief rank by z instead of by FDR survival?

## 4. Did the candidates use it?

**Delivery:** all 16 candidate dirs, repairs included, have a `critiques.md` byte-identical
to their round's `critique/critiques.md`.

**Use** (my qualitative read of each `hypothesis.md`):

| round | slot | name | outcome | cites a discrepancy? |
|---|---|---|---|---|
| 1 | 1 (explore) | `trembling_hand_speaker` | rejected (ELPD not finite) | yes: foils on ambiguous words |
| 1 | 1 repair | `trembling_hand_speaker_2` | admitted | yes (same text) |
| 1 | 2 (explore) | `distractor_elimination_speaker` | admitted | yes: singleton when the word fits all (twins) |
| 1 | 3 (explore) | `lexical_accessibility_speaker` | admitted | weakly ("negative implicatures") |
| 1 | 4 (incumbent) | `r1_c4` | admitted, **became incumbent** | no (see below) |
| 1 | 5 (incumbent) | `competitor_confusion_rsa_l2` | rejected (not novel vs `r1_c4`) | yes: "curbs the over-prediction of intermediate scalar competitors" |
| 1 | 5 repair | `costly_extension_rsa_l2` | admitted | yes: singleton when the word fits all |
| 1 | 6 (chosen) | `isolated_graded_costly_singleton_l3` | admitted | no |
| 2 | 1 (explore) | `communicative_expressibility_listener` | admitted | no |
| 2 | 2 (explore) | `cognitive_hierarchy_listener` | admitted | no |
| 2 | 3 (explore) | `cooccurrence_lexical_uncertainty_l2` | admitted | yes: foils on ambiguous, accuracy on unambiguous |
| 2 | 4 (incumbent) | `trembling_hand_confusion_rsa_l2` | admitted | yes: the same pair |
| 2 | 5 (incumbent) | `trembling_hand_confusion_l2` | rejected (not novel) | yes: the same pair |
| 2 | 5 repair | `costly_extension_confusion_rsa_l2` | admitted, **exported** | weakly ("overly broad expression") |
| 2 | 6 (chosen) | `trembling_confusion_speaker` | rejected (not novel) | yes: foils |
| 2 | 6 repair | `isolated_graded_costly_feat_l3` | admitted | yes: feature-rich prior (`size_prior…`) |

**Count:** 10 of 16 attempts cite a discrepancy explicitly, 2 weakly, and 4 not at all. The
statistics cited most are the two with the largest z (the foil rate and unambiguous
accuracy).

**A side finding: `r1_c4`'s hypothesis is stale.** This is the round-1 incumbent refinement
that won round 1:
- Its code adds competitor-confusion aversion (`w_confusion`, `confusion = real_lex @ sim`),
  and its docstring says so.
- But its `hypothesis.md` is the parent's first sentence, verbatim (23 words), and it wrote
  no `model_name.txt`.
- So the ledger, `history.json` and round 2's critique context and briefs described the
  incumbent without its new mechanism.

Should admission require a hypothesis that differs from the parent's? That is for the driver.

## 5. Time and spend

**Time** (from file times and token-record timestamps; the log has no timestamps):

| phase | PDT | duration |
|---|---|---|
| design (EIG with quotas) | 14:49 – 15:37 | 48 min |
| collect, CV folds, seed fits | 15:37 – 15:56 | 19 min |
| **critique 1**: agent 13.2 min, scoring 3.7 min | 15:56 – 16:13 | **16.8 min** |
| round 1 candidates, repairs, admission | 16:13 – 17:45 | 91 min |
| **critique 2**: agent 28.0 min, scoring 3.3 min | 17:45 – 18:16 | **31.3 min** |
| round 2 candidates, repairs, admission, end-of-experiment prune, export | 18:16 – 20:05 | 109 min |
| **total** | 14:49 – 20:07 | **5 h 18 min** |

The critique took 48 of 318 min (15%). Most of that is the agent; the 1,000-replicate scoring
is 3-4 min. The handoff's "a few minutes" is too low. Budget about 15-30 min a round.

**Spend** (`token_usage_summary.json`):

| | calls | tokens | cost |
|---|---|---|---|
| `rsa:candidate` | 16 | 151.0 M | $19.74 |
| `rsa:critique` | 2 | 32.5 M | **$4.04 (17%)** |
| total | 18 | 183.5 M | $23.78 |

A critique call costs $1.92 and $2.12. A candidate call costs $1.23 on average ($0.43–1.73).
There were no usage-limit hits.

## 6. The design

`experiment1/design/eig.json`:

**Pool and models:**
- `pool_size` **1,597**, and `screened_out` is **empty**;
- 6 carried and 7 bar models, with `n_withheld` 0.

**Quotas: all met** (count ≥ minimum):

| quota | minimum | count |
|---|---|---|
| objects=2 | 6 | 6 |
| objects=3 | 10 | 10 |
| query=prior | 8 | 13 |
| query=word | 20 | 27 |

**Kinds:** 2×3 prior 1, word 1; 2×4 prior 3, word 1; 3×3 word 1; 3×4 prior 3, word 6; 4×3
prior 1; 4×4 prior 5, word 18.

**The free design** (no quotas, recorded beside it) has 1 two-object display and 11
three-object displays. So the quotas move 5 picks onto 2-object displays, 4 of them from
4-object displays.

**Power** at N=200 (50 responses a display):

| | joint EIG | p(correct) | mean posterior on truth |
|---|---|---|---|
| with quotas | 3.445 of 3.696 bits | **0.933 ± 0.006** | 0.899 |
| free | 3.450 bits | 0.931 ± 0.006 | 0.900 |

The quotas cost no power.

## 7. As before

**The ground truth in agent outputs:** `gt_name_mentions.txt` is empty. The critique dirs are
inside `round_*`, so they were covered.

**Recovery** (`private/experiment1/recovery.json`; KL is now a mean per display):

| | closest before: `rsa_l2_singleton_feat_color_valence_l0` | `r1_c4` | **exported** `costly_extension_confusion_rsa_l2` |
|---|---|---|---|
| `kl_pool` (mean, 1,597 displays) | 0.0168 | 0.0022 | **0.0020** |
| `kl_design` | 0.0475 | 0.0046 | 0.0041 |
| `rmse_pool` | 0.056 | 0.020 | **0.019** |

- **The distance fell 8.5-fold**, nearly all of it in round 1. That is `r1_c4`'s
  competitor-confusion term, the mechanism closest to the ground truth's "confusion".
- **The same caveat as rehearsal 2 applies** (`REHEARSAL2_REPORT.md` §5): the ground truth
  descends from these seeds, so this is not a recovery test.
- Rehearsal 2's experiment 1 reached RMSE 0.006 after 5 rounds on a different pool; it is
  not comparable with this run's 2 rounds.

**The live set exported:** `costly_extension_confusion_rsa_l2`, `r1_c4` and
`isolated_graded_costly_singleton_l3`.

**Ledger:**
- 12 admitted.
- 4 rejected: 1 for non-finite ELPD (repaired) and 3 as not novel.
- 15 pruned at the end of the experiment:
  - 4 ineligible by the guard (more than 4 clustered SE behind on the other trials);
  - 11 more than 2 clustered SE behind the best on the live trials' grouped CV.

**Prospective score** (experiment 1, no committed model yet):
- The best live model beat the best seed on the new data by **115.6 ± 55.1 nats**.
- It was 0.49 ± 0.19 behind the best promoted model.

**Crashes:** none. The log's only non-deprecation warnings are two numpy "invalid value
encountered in divide" lines. The one candidate whose ELPD came out NaN was rejected and
repaired.

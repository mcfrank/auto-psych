# pragmods reference-game data (seed dataset)

Four external datasets in the same schema (Mayn & Demberg 2022/2023/2026,
Sikos et al. 2021) are documented at the end, under "External reference-game
datasets".

`pragmods_trials.csv` is every response in the de-identified MTurk data of the
pragmods experiments (Frank, Emilsson, Peloquin, Goodman & Potts, "Rational
speech act models of pragmatic reasoning in reference games"; data
github.com/langcog/pragmods, MIT licence) as one trial-level table: 10,168 rows
from 7,965 participant records (8,569 rows included by the analyses' own
exclusion criteria; a person who took part in several experiments has one
record in each).

Regenerate it from a clone of langcog/pragmods:

```bash
uv run python -m src.rsa.pragmods_ingest --pragmods-dir <clone of langcog/pragmods>
```

`pragmods_trials.provenance.json` records the pragmods commit it was built
from, every source file with its row count and sha256, the command, and the
CSV's sha256. `reference/` holds `models.csv`, `prior_counts.csv`, `levels.csv`
and `prior.csv`, copied unchanged from pragmods `models/data/`, which the
tests (`tests/test_pragmods_ingest.py`) reproduce from the CSV. With
`PRAGMODS_DIR=<clone>` the tests also regenerate the CSV and require it to be
byte-identical to the committed one.

Read: every participant file under pragmods `data/` except `unused/`,
`4-sequences/originals/` and `3-levels/size/unused-size/` (the ingest raises if
a file is neither read nor excluded). Not carried: free-text comments,
"about" answers, age, gender, HIT metadata, MTurk assignment status. The only
free text in the CSV is the speakers' own descriptions (the dependent variable
of the production experiments, in `response`).

## Columns

JSON values are encoded as strings. "Row" below means a matrix row (object);
"column" a feature.

| column | meaning |
|---|---|
| `participant_id` | anonymous id (`S` + 5 digits), the same person across files. For the `4-sequences` files and `5-speakers/pragmods_overspec_baseline`, whose ids were within-file row numbers before de-identification, it is prefixed with the batch (`pragmods_seq:S12345`): those cannot be linked to other files. |
| `batch` | the batch's 4+-letter code (`ALLS`, `FAMO2`, ...) or, without one, the file stem |
| `source_file` | path relative to pragmods `data/` |
| `series` | its folder, e.g. `3-levels/levels` |
| `experiment` | `E1_dv` ... `E10_oddman` (the paper's experiments 1-10, Table 1), `color_prior_rerun`, `levels_prior_action`, `size`, `sequences`, `speakers` (see below) |
| `condition` | the condition within the experiment (labels below) |
| `paper_cond` | the cell of the paper's model data the row is counted in: `models:<expt>/<cond>` (a `models.csv` cell) or `prior_counts:<prior>` (a `prior_counts.csv` row); empty for rows in neither |
| `trial_index` | 0 for one-shot experiments; 0..k within a participant's sequence |
| `item` | friend, boat, pizza, snowman, sundae, Christmas tree |
| `feature_names` | JSON list, the item's word for each column, in column order; `null` where the data do not record which word that column was (feature-to-word assignment was randomised per participant) |
| `objects` | JSON list of 0/1 lists, one per object, in the experiment code's canonical (unpermuted) row order |
| `object_roles` | JSON list, the code's role label for each object (`foil`/`target`/`logical`; `single`/`twin`; `twin_1`/`twin_2`/`odd_one`; size: `target`/`distractorK`/`refK`) |
| `display_order` | JSON list, object index at each screen position left to right; `null` at a position the data do not pin down; `[]` when nothing is recorded |
| `query` | `utterance` (the listener is given a feature word, or the colour patch of one), `prior` (no informative message: mumble, "which will he X next", silent favourite), `production` (speaker trials) |
| `query_detail` | the specific variant (e.g. `mumble_one_word: ...`, `which_next: ...`, `color_patch_pointing`, the sequence level) |
| `utterance` | column index of the uttered feature; empty for prior and production |
| `framing` | label of the `linguistic_framing` code (`one_word`, `my_favorite_X_has`, `my_least_favorite_X_has`, `my_X_has`, `silent_favorite`, `points_to_color_patch`, ...) |
| `familiarization` | JSON per-object counts out of 9 familiarization images (E5 only), else empty |
| `grayscale` | JSON per-object 1 = shown in greyscale (E7 and its rerun only), else empty |
| `dv` | `forced_choice`, `betting`, `likert`, `production` |
| `choice` | chosen object index (forced choice) |
| `response` | betting: JSON dollars per object (sum 100); likert: JSON 1-7 per object; production: JSON `{"modality", "description"` or the four checkbox booleans`, "overspec"}`; the speaker level-1 listener trials (no matrix): `{"choice_role": ...}` |
| `referent` | production: index of the object the speaker had to describe |
| `included` | `True` when the participant passes the exclusion criteria of the analysis Rmd that reads the file |
| `exclusion_reason` | `;`-joined: `manip_check_target`, `manip_check_dist`, `name_check`, `duplicate_participant`, `overspec_missing` |
| `notes` | `;`-joined flags for anything inferred or unrecorded on that row (listed under Known gaps) |

## Rows per experiment

participants (included) / rows (included) / condition rows (included):

| experiment | series | participants | rows | conditions |
|---|---|---|---|---|
| E1_dv | 1-prelims/dv | 689 (554) | 689 (554) | forced_choice 290 (248); betting 199 (158); likert 200 (148) |
| E2_manip_check | 1-prelims/manip | 580 (513) | 580 (513) | manip_check 340 (306); no_manip_check 240 (207) |
| E3_ling_frame | 1-prelims/linguistic_framing | 100 (89) | 100 (89) | one_word 57 (51); my_X_has 43 (38) |
| E4_prior_frame | 2-prior/measurement | 200 (175) | 200 (175) | mumble 104 (88); action 96 (87) |
| E5_baserate | 2-prior/baserates | 800 (488) | 800 (488) | inference_baserate_{0.11,0.33,0.44,0.77} 100 (72), 93 (63), 107 (68), 100 (75); prior_baserate_{...} 96 (53), 103 (57), 99 (49), 102 (51) |
| E6_valence | 2-prior/language | 550 (478) | 550 (478) | favorite 200 (176); least_favorite 100 (86); favorite_prior 116 (100); least_favorite_prior 134 (116) |
| E7_color | 2-prior/color | 300 (264) | 300 (264) | inference_color_{none,foil,logical,target} 42 (38), 53 (47), 54 (48), 51 (46); prior_color_{foil,logical,target} 31 (27), 27 (24), 42 (34) |
| color_prior_rerun | 2-prior/color | 100 (83) | 100 (83) | prior_color_{foil,logical,target} 34 (30), 30 (24), 36 (29) |
| E8_levels | 3-levels/levels | 416 (345) | 416 (345) | simple_L0 65 (49); simple_L1 55 (46); complex_L0 62 (52); complex_L1 65 (58); complex_L2 73 (62); complex_prior 96 (78) |
| levels_prior_action | 3-levels/levels | 104 (91) | 104 (91) | complex_prior_action 104 (91) |
| E9_twins | 3-levels/twins | 220 (193) | 220 (193) | twin 49 (46); uniform 71 (60); prior 100 (87) |
| E10_oddman | 3-levels/oddman | 300 (269) | 300 (269) | patch 102 (93); word 98 (89); prior 100 (87) |
| size | 3-levels/size | 1750 (1368) | 1750 (1368) | 13 inference conditions `inference_<N>obj_<F>feat_sl<k>` (1300 rows, 985 incl.); 9 prior matrices `prior_<N>obj_<F>feat` (450, 383) |
| sequences | 4-sequences | 500 (477) | 1800 (1710) | 1w0w1b 345 (333); 0w1w1b 255 (240); (0w1w)x3 300 (282); (0b1b)x3 300 (276); 0w1w2w 138 (129); 2w1w0w 162 (159); 0w2w1w 135 (135); 1w2w0w 165 (156) |
| speakers | 5-speakers | 1356 (1166) | 2259 (1949) | L1:{text,checkbox,virtual_keyboard} 149 (121), 150 (136), 154 (126); seq_L1:{...} 302 (264), 300 (254), 304 (270); seq_L2:{...} 300 (270), 300 (272), 300 (236) (seq rows = 2 per participant) |

## How the data reproduce the paper

* **models.csv / prior_counts.csv.** Aggregating the included forced-choice
  rows of each `paper_cond`, mapping each chosen object to the
  `models/matrices.R` row with the same features (a choice of one of two
  identical twins counts half for each, as models.csv does), reproduces every
  row of `prior_counts.csv` and every `n` and every count of `models.csv`
  except two cells, which models.csv labels differently from the stimuli:
  * `complex` cond `1` ("mustache"): the data are 44 M, 11 GM, 3 HG.
    models.csv has `foil`(M) 44, `logical`(HG) 11, `target`(GM) 3, i.e. it puts
    the 11 literally-true GM choices on HG, which has no mustache.
  * `simple` cond `0` ("hat"): the data are 48 HG, 1 featureless face, 0 G.
    models.csv has the 1 on `target`(G) and 0 on `foil`.
  Both come from the hand relabelling in `models/data/levels_mod.csv`, which
  gave each of these cells the other's permutation. levels.csv and prior.csv
  (the Rmds' own output, by role label) are reproduced exactly, all cells. For
  the comparison, the CSV's columns are matrices.R's `hat, glasses` (simple),
  `hat, glasses, mustache` (complex, twins) and `hat, mustache, glasses`
  (oddman, so that twin_1 is matrices.R's `logical` and twin_2 its `foil`, as
  levels.Rmd maps them).
* **Table 1.** N_total matches for all ten experiments. N_include matches for
  E1-E5. For E6-E10 Table 1 counts participants passing the manipulation and
  name checks *before* repeat participants are removed: 502 = 478 + 24,
  267 = 264 + 3, 362 = 345 + 17, 194 = 193 + 1, 270 = 269 + 1 (the second
  term is the participants excluded only as duplicates). The analyses and
  models.csv do remove them, and so does the CSV.
* The commented-out rows of the table (size 1750/1368, sequences, production)
  also match, with two labels swapped there: "Level 1 x3 200/193" is the two
  level-2 sequence files and "Level 2 100/93" the two x3 files; the production
  "Level 1" N_total of 450 is nominal (the files hold 453).

## Coding decisions

**Matrices.** Every one-shot listener display is one `scale_and_level`
condition of `pragmods_parameter_setter_c1.js` (langcog/pragmods-expts
`8ff10c3`), transcribed in `C1` in the ingest. `objects` are its unpermuted
`expt` rows; a column present on no object (the third feature of the simple
matrix, never shown on any object) is dropped.

| scale_and_level | matrix | objects (roles) | uttered column |
|---|---|---|---|
| 0 | simple, level 0 | 00 foil, 01 logical, 11 target | col 0 (only on target) |
| 1 | simple, level 1 (the standard game) | 00 foil, 01 target, 11 logical | col 1 |
| 2-4 | complex (M, GM, HG) levels 0-2 | 001, 011, 110 | 2: col 0 (HG); 3: col 2 (M); 4: col 1 (GM) |
| 5-7 | twins | 011 single, 101 twin, 101 twin | 5: prior; 6: col 0; 7: col 2 (on all) |
| 8 | oddman | 011 twin_1, 101 twin_2, 110 odd_one | col 2 |

The 2013 files (E1, the E6 favourite batches, LEAS) record no
`scale_and_levels_condition`; `all_experiments.csv` gives 1 for all of them
(note `scale_and_level_1_from_all_experiments_csv`).

**Display and feature names** are solved jointly from whatever a file records
(`target_position`, `logical_position`, `{target,logical,foil}_property`,
`left/middle/right_items`, `position_chosen`/`items_chosen`,
`position_with_color`): every layout of the matrix consistent with all of it is
enumerated, and a position or word is filled only where all agree. A file
whose records admit no layout raises. Layouts that differ only by swapping
identical objects (twins) or identically distributed features are one layout:
identical objects are placed in index order left to right. The c1 code's
`logical_position` is the position of its "distractor", which is always the
`logical` row.

**Queries.** `question_type` 0 with a word (`linguistic_framing` 0-6, 10, 13,
14) or the colour patch (9) is `utterance`; 0 with framing 7/8 (silent
favourite / least favourite), 1 and 3 (mumble), 2 and 4 are `prior`.

**Exclusions** follow each analysis block of the Rmds: the manipulation checks
(`manip_check_target`/`_dist` against the counts each condition requires),
the name check, and `duplicated(workerid)` over the block's bound files in the
Rmd's order before any other filter (so a repeat is excluded even when the
earlier record failed a check). E1 and E3 compare as strings, the rest via
`as.numeric`, as the Rmds do. E2 accepts -1 (not asked). levels.Rmd's
per-condition answers: simple L0 (1, 2), simple L1 (none, 1), complex L0
(1, 2), L1 (2, 2), L2 (2, 1), complex prior (2, 1); twins prior (1, 2), twin
(2, 1), uniform (3, 1); oddman (2, 2). Size: name check and duplicates over
the four size files. Sequences: name check only (no deduplication, as
Sequences.Rmd). Speakers: name check and a coded `overspec`.

**Condition labels.** E5 baserate labels are models.csv's `0.11, 0.33, 0.44,
0.77` for 1/9, 3/9, 5/9, 7/9 of the familiarization images showing the
pragmatic target (5/9 is labelled 0.44 there, sic); `familiarization` has the
counts (foil, target, logical) 1/1/7, 1/3/5, 1/5/3, 1/7/1. E7's condition is
the referent shown in colour.

**Experiments outside Table 1.** `color_prior_rerun` (COLORBASE2) is a
second colour prior batch that prior.Rmd does not read; `levels_prior_action`
is the `question_type` 4 ("which will Bob X next") half of SCALESBASE, which
levels.Rmd drops; both are coded and excluded like their siblings (deduplicated
within themselves) and have no `paper_cond`. `size`, `sequences` and
`speakers` are analysed in the Rmds but commented out of the paper.

## Known gaps (flags in `notes`)

* `color_none_condition_...` — E7 "none": the c1 code documents colour
  condition 0 as "all elements in colour"; the paper says "none of the
  referents was shown in colour". The CSV follows the code (`[0,0,0]`); either
  way no referent is singled out.
* `display_order_partial`, `feature_names_partial` — the file records only
  some positions / words (most often only the target's).
* `choice_is_one_of_identical_objects` — a choice of "twin" (WERD, which has no
  chosen position) or of a size condition's two identical targets; the lower
  index is given.
* `sequence_complex_L0_logical_role_inferred_from_distractor` — the
  sequence experiments ran a later code version. Its recorded words and
  positions fix every display (levels 0/1 of seq, wx3 and bx3 on one simple
  display; levels 0-2 of seq2 and L2second on one complex display), and agree
  with the c1 coding except at complex level 0, where its "distractor" is M
  (mustache column), not GM. Since the c1 distractor is always the `logical`
  object, M is taken as `logical` there; this only decides which of the two
  literally false objects the 10 level-0 non-target choices were.
* `distractor_prop_recorded_equal_to_target_prop`,
  `distractor_prop_inconsistent_ignored` — some versions recorded the target
  word as the distractor's (the listener trials of the text level-1 and all
  level-2 speaker files, and L2second "0w2w1w" trial 2) or a distractor word that contradicts the other
  trials (L2second "0w2w1w"); those words are ignored, the positions agree.
* `size_distractor_numbering_inferred` — the size inference files record a
  choice of `target` or `distractor1..3` but not which object each distractor
  is. They are numbered here as the non-target rows in matrices.R order (by
  feature count). Evidence: `distractor3` is never chosen where there are
  only two non-targets, and where there are three the logical (most-featured)
  object is `distractor3` (e.g. 4x2 "3, glasses": 0/0/19). Exact for 2-object
  matrices. No display order is recorded.
* `utterance_is_one_of_identical_feature_columns` — size conditions where the
  uttered word is one of several columns with the same extension.
* `slot_order_object_1_to_n_assumed_left_to_right` — the size prior file
  records the objects as `object_1..4_items`; the matrix is matched to them and
  the chosen object comes from `position_chosen`/`items_chosen`. levels.Rmd
  warns its `choice` labels are unreliable, so they are not used.
* `speaker_display_simple_matrix_from_features_in_referent_0_1_2` — speakers
  described an object with 0, 1 or 2 features, which only the simple matrix
  has. Words are not given: the recorded property names disagree between code
  versions. Display order is recorded only for the sequential files
  (`display_order_from_target_distractor_positions_unverified` where
  `position_to_describe` is missing to check it).
* `speaker_seq_listener_matrix_complex_because_level_2`,
  `speaker_seq_level1_listener_matrix_unknown` — the listener trial after a
  sequential production. Level 2 exists only in the complex matrix (target
  GM, word "glasses"). The level-1 trial's matrix (simple or complex) is not
  recorded: `objects` are empty and `response` holds the chosen role.

## Notes for modelling

* One judgment per participant in E1-E10 and size; repeated measures only in
  `sequences` (3 or 6 listener trials, same display repeated or a new one per
  `sequence_condition`: digits are the trial levels in order) and the
  sequential speaker files (a production then a listener trial).
* Manipulations of the prior: familiarization base rates (E5), valence of the
  framing word — favourite / least favourite, inference and silent prior (E6),
  greyscale salience (E7), and the matrix itself (levels, twins, oddman, size);
  most inference conditions have a matching prior condition (`query=prior`).
* Size: matrices 2-4 objects x 2-4 features (`models/matrices.R`
  `size{N}.{F}`), the target the least-featured object with the word.
* E1 betting and Likert responses are per object; forced choice elsewhere.
* The display was randomised per participant (`display_order`) and so was the
  item's feature-to-word assignment (`feature_names`).

# External reference-game datasets (`src/rsa/ingest/`)

Four more sources in the same trial schema, built by
`src/rsa/ingest/` from pinned public files (URL + sha256; a changed or
corrupt download raises):

```bash
uv run python -m src.rsa.ingest.run --sources mayn_demberg_2026 sikos_2021 mayn_demberg_2023 mayn_demberg_2022
uv run python -m src.rsa.ingest.combine --sources pragmods mayn_demberg_2026 sikos_2021 --out <combined.csv>
```

Downloads are cached in the gitignored `data/rsa/external/<source>/raw/`.
**What is committed depends on the licence** (PI decision): the CC-BY sources'
CSVs (`mayn_demberg_2026_trials.csv`, `sikos_2021_trials.csv`) are committed
here; the sources whose repositories carry no licence (`mayn_demberg_2023`,
`mayn_demberg_2022`) are written only to
`data/rsa/external/<source>/<source>_trials.csv`, and only their pin file is
committed. Every source has a `<source>_trials.provenance.json` here: citation,
licence, the pinned files (URL, bytes, sha256), the GitHub commit where there
is one, the command, and the derived CSV's row/participant counts and sha256.
The tests (`tests/test_rsa_ingest_*.py`) need nothing but the committed files;
those about the uncommitted sources, and the byte-for-byte rebuilds, skip
unless the cache is built or `RSA_INGEST_FETCH=1` lets them fetch.

`combine` concatenates any of the sources (pragmods included) and checks
that each meets the column contract and that experiment names and participant
ids do not collide. `src.rsa.dataset.load_forced_choice` reads any single
derived CSV or the combined one.

## Column contract of the external sources

The pragmods columns above, with three more:

| column | meaning |
|---|---|
| `source` (first) | `mayn_demberg_2026`, `mayn_demberg_2023`, `mayn_demberg_2022`, `sikos_2021` (`pragmods` in a combined file) |
| `messages` (after `notes`) | JSON list of the feature indices the speaker could name on the trial (`Context.messages`). The heard `utterance` is always one of them and true of some object. Empty for pragmods (every feature is a word). |
| `covariates` (last) | JSON object of per-row extras (below); `{}` for pragmods |

Participant ids are `<source>:<id in the source file>` (Sikos:
`sikos_2021:e<k>:<row>`), experiment names carry a source prefix, so neither
can collide across sources. `familiarization` and `grayscale` are empty. No
row is an excluded participant's *dropped* data: every response is kept and
`included` / `exclusion_reason` say whether the paper analysed it.

## mayn_demberg_2026, mayn_demberg_2023, mayn_demberg_2022 (Franke & Degen game)

* **2026** — Mayn & Demberg, "Sources of individual variability in a pragmatic
  reference game: Effects of logical reasoning and Theory of Mind", PLoS One
  21(2): e0339899. Data OSF 5ab3f (`data/main_task_data.csv`,
  `ID_scores_with_exclusions.csv`, `annotations.csv`), **CC-BY 4.0**. Shapes
  and colours, Prolific. 306 participants x 66 trials = 20,196 rows; 254
  included (16,764 rows). Experiment `md2026_shapes`.
* **2023** — Mayn & Demberg, "High performance on a pragmatic task may not be
  the result of successful reasoning", Open Mind 7: 156-178.
  github.com/sashamayn/refgame_stimuli_methods @ `d7d4aec`
  (`data/all_experiments.csv`, `data/all_annotations.csv`), **no licence**.
  237 participants (59/59/59/60), 15,642 rows; 228 included (57/55/56/60).
  Experiments `md2023_e1_replication` (monsters), `md2023_e2_remapped`,
  `md2023_e3_all_messages`, `md2023_e4_shapes`.
* **2022** — Mayn & Demberg, "Individual differences in a pragmatic reference
  game", CogSci 44: 3016-3022. github.com/sashamayn/refgame_cogsci22 @
  `b3a2b5e` (`main_task_data.csv`), **no licence**. 115 participants (pilot
  47, main 68; `batch` = the file's `study`), 7,590 rows, all included.
  Experiments `md2022_pilot`, `md2022_main`.

**Game.** Each of 66 trials shows three objects, each one value on each of two
dimensions, and a message naming one feature, said to be sent by an earlier
participant; the listener clicks the object meant. 24 critical trials (12
simple, 12 complex implicatures), 9 completely ambiguous fillers (two
identical objects the message is true of) and 33 unambiguous fillers (types a,
b, c and "filler unambiguous"). Some features have no message, which is what
makes the critical trials solvable. Items are fixed (same display for every
participant); their order and the screen layout are randomised. All rows are
`dv=forced_choice`, `query=utterance`, `framing=message_icon` (a new framing
label, valence 0, in `src/rsa/dataset.py`: the message is shown as a picture
of the feature, a shape contour or colour tube / a creature or accessory
alone; `query_detail=picture_of_feature`).

**Features.** A fixed six-column vocabulary per stimulus family (`feature_names`),
ordered so that the families correspond under the 2023 paper's Exp. 4 mapping
(square = robot, triangle = green monster, circle = purple monster, blue =
scarf, green = red hat, red = blue hat):

| col | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| shapes | circle | triangle | square | green | red | blue |
| monsters | purple_monster | green_monster | robot | red_hat | blue_hat | scarf |

Columns present on no object of a display are kept (they are not words of the
context: `Context.present_features` drops them).

**Message sets** (`messages`), confirmed from the papers and from the messages
the data contain: the original game (2026, 2022, 2023 Exps. 1 and 4) `[0,1,3,4]`
(no square/robot, no blue/scarf); 2023 Exp. 2 "remapped" `[1,2,3,5]` (robot and
scarf nameable, purple monster and blue hat not; the trials are Exp. 1's with
those images swapped, and the data record the images shown); 2023 Exp. 3 "all
messages available" `[0..5]` (same trials and messages heard as Exp. 1, but the
speaker could have named every feature, so the simple trials are no longer
solvable by reasoning). Exps. 1 and 4 have identical displays per item (a test
checks).

**Roles and choices are derived from features, not from the label columns**,
which are wrong in places: `targetpos` is wrong for item 20 (and for some
ambiguous fillers) in 2026; the `target`/`distractor` labels of item 20 are
swapped in all four 2023 experiments and in the 2022 main study (whose
`correct` is then wrong too), the `target`/`competitor` labels of item 51 in
all of 2022; `answer_which` is arbitrary when one of the two identical objects
of an ambiguous filler is chosen. Objects are in role order: for a critical
trial the target is the object the message is true of that cannot be named
alone and the competitor the one that can (it has a nameable feature no other
object has), using the message set the items were designed for (the original
one for 2023 Exp. 3); ambiguous fillers are `twin_1, twin_2, distractor`;
unambiguous fillers `target` (the only object the message is true of) and the
labelled competitor and distractor. A row whose labelled target is not the
feature target is flagged `labelled_target_is_not_the_feature_target` in
`notes` (2023: 237 rows, item 20; 2022: 183 rows, items 20 and 51; 2026:
none). `display_order` is the screen layout from `presentation_order`
(identical objects in index order left to right); `choice` is the object at
`answer_order`'s position, checked against `answer`. `trial_index` is the
trial's position in the participant's sequence (`trialid`, 0-based); `item` is
`<family>_item<itemid>`.

**Exclusions.**

* 2026, as the paper (N = 254): of the 300 first-session participants, 8 with
  accuracy < 0.8 on the 33 unambiguous fillers (`unambiguous_accuracy_below_0.8`;
  the file's `main_exclude`, checked against the accuracy computed from the
  derived roles), then anyone whose strategy explanation for either re-shown
  item is annotated `misunderstood_instr` (23 of the 292) or `odd_one_out`
  (15 more; `strategy_*`). Participants 301-306 are in the trial file but
  not in the individual-differences file and so not among the paper's 300
  (`not_in_paper_sample`); the analysis scripts' `main_exclusions()` cannot
  exclude them by accuracy and keep 259 (254 + 5 of them; a test checks
  both). They pass the accuracy criterion (0.91-1.0); one is also
  strategy-excluded.
* 2023, as `stat_helpers.R` and Table 1 (57/55/56/60): accuracy < 0.8 on the
  33 non-ambiguous fillers, computed from the derived roles (equal to the
  file's `correct` on these trials).
* 2022: none. The scripts exclude no one, every participant reaches 0.8 on
  the unambiguous fillers, and the 2026 paper cites the main study's N = 68.
  The CogSci paper itself could not be retrieved (escholarship blocks
  automated access), so its pilot N (47) is the data's, unverified.

**Covariates.** 2026 and 2023: `strategy_tag_simple`, `strategy_tag_complex`,
the participant's annotated reasoning strategy (`tag_both`) for the re-shown
simple and complex item (correct_reasoning, guess, other_reason, unclear,
misunderstood_instr, odd_one_out, exclude, ...). The explanations' free text is
not carried. 2022: none.

**Reproduced from the papers** (`tests/test_rsa_ingest_mayn_demberg.py`):
2026 — 300 / 8 / 292 / 23 / 15 / 254; 148 and 100 of the 292 tagged
correct_reasoning (simple, complex); 70% of the included participants on or
below the diagonal of Fig 3. 2023 — Ns 57/55/56/60; the annotation exclusions
the text reports (Exp. 1 simple 1 exclude + 8 unclear; Exp. 2 simple 4 + 6,
complex 3 + 10; Exp. 3 simple 1 + 7, complex 0 + 6). 2022 — 47 + 68.

**Known gaps.**

* 2026 guess counts: the paper says 73 (simple) and 129 (complex) of 292; the
  published annotations give 72 and 130. Its "distractor on 1.2% of trials"
  (critical + ambiguous trials, N = 254) comes out at 1.7% here (2.0% on
  critical trials only); not resolved.
* The re-shown item at the end (2023 `strategy_*` columns, the 2026
  annotations' `answer`) is a second response to an item already answered,
  given with a written explanation and possibly a new layout; it is **not** a
  trial here. Its annotation tag is kept as a covariate.
* The 2026 individual-difference scores (CRT, Raven, digit span, OSpan, SST,
  RMET; CC-BY) and the 2022 `averages_with_indiv_measures.csv` are not
  carried; join them on the source id if a model needs them.
* The size of `mayn_demberg_2026_trials.csv` is ~9 MB (20,196 rows).

## sikos_2021

Sikos, Venhuizen, Drenhaus & Crocker (2021), "Reevaluating pragmatic reasoning
in language games", PLoS One 16(3): e0248388, **CC-BY 4.0**; the article's
supporting files S1-S3 (`exp{1,2,3}_data`). A one-shot Frank & Goodman (2012)
replication on MTurk: one trial per participant, so one row each (`batch` =
task, `trial_index` 0). 7,488 rows; 5,625 included.

* **Displays.** Three objects, each a colour and a shape (`orange.fish`).
  Columns are the display's colours, then its shapes, by first appearance left
  to right; every feature is a word, so `messages` lists every column.
  `objects` are `o1, o2, o3`, left to right (`display_order` `[0,1,2]`; the
  paper puts the speaker's target, `targ`, in the middle in Exps. 1 and 3).
  `task.resp` A/B/C is o1/o2/o3 (this reading makes 98% of all listener
  choices literally true, the reverse 77%). `object_roles`: `target` (`targ`) and
  `other`; Exp. 2 names its `colour_competitor` and `shape_competitor`.
* **Tasks.** Listener (`query=utterance`, `listen.word`), Salience (Robert says
  something incomprehensible; `query=prior`), both `dv=forced_choice`;
  Speaker (Exp. 1 only; `dv=production`, `query=production`): `response` is
  `{"word", "feature", "options"}` (the chosen word, its column, the two words
  offered: the target's colour and shape), `referent` the target.
* **Experiments.** `sikos2021_e1` (24 context types, iconic objects; speaker,
  listener, salience), `sikos2021_e2` (the 2s2c.b context; iconic vs
  geometric stimuli, `item` and covariate `stimulus_type`; listener, salience),
  `sikos2021_e3` (Exp. 1's 8 RSA-diagnostic contexts with a more engaging
  cover story; listener, salience). `condition` is the context code (`cond`,
  Exp. 3 `context`), whose last letter is the listener's word type (c/s).
* **Exclusions**, reproducing every N in the paper (counted in its order):
  `non_native_or_non_fluent` (`language`, Exp. 2 `nativeLang`, must be
  English and `fluency` fluent), `attention_check_failed` (`attnQ.acc`),
  `listener_choice_not_literally_true`. Exp. 1: 4642 recruited, 1137 / 118 / 13
  excluded, kept speaker 1143, listener 1098, salience 1133. Exp. 2: 1671;
  142 / 77 / 12; listener 960, salience 480. Exp. 3: 1175; 265 / 96 / 3;
  listener 405, salience 406. Exp. 2 uses `task.resp` (not `task.resp1`).
* **Covariates.** `display` (the item id), `side` (Exps. 1, 3; `a`/`b`,
  undocumented, plausibly the paper's mirrored-context counterbalancing; the
  objects are in screen order either way), `stimulus_type` (Exp. 2), `task_likelihood`
  (Exp. 1's 0-100 rating after the choice; not documented in the article).
* Dropped: demographics, browser/OS/screen strings, timings, `exit.*`, survey
  answers and all free text. There is no participant id in the files
  (`server.intern.id` repeats), so the id is the row number.

## Not ingested

* **Duff, Mayn & Demberg (2026)**, "The role of reinforcement learning in
  pragmatic reasoning tasks", Open Mind (OSF 7uwx9 / ad685; no licence). Its
  method section: "we provided participants with feedback after each trial,
  indicating whether their response was the intended target", after a speaker
  pre-training; the ambiguous fillers were also removed. Its listener choices
  are therefore learned under trial-by-trial reinforcement, not
  interpretations of the kind the other sources measure, and it is left out
  (`NOT_INGESTED` in `src/rsa/ingest/run.py`).
* Excluded by the PI: Franke & Degen (2016); any child data; the imagined-child
  and ChatGPT-speaker conditions (OSF f5nmv, perceptions_of_chatgpt); slider
  studies (OSF erbn3); the listener-adaptation/feedback study (OSF 5d2f6);
  Franke, Tsvilodub & Carcassi (2024).

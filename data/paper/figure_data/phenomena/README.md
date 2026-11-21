# Patterns the winning models predict and every seed model misses

These files hold the data for figures showing *why* the auto-psych winners fit better than the
seed models they started from. Each file has one row per human trial. Each row records what the
person did and what each model predicted for that trial.

A pattern counts as one "every seed misses" only if two conditions hold:
- **The seeds miss it.** Every individual seed misses it, and so does a **seed blend**: a
  cross-validated combination of all the domain's seeds, the strongest baseline the seeds can give.
- **The winners get it.** The winning models match what people do.

So in each figure, compare **people** with the **winners**, the **seeds** and the **blend**.

## The six patterns

### Subjective randomness (`sr_trials.csv`)

People see two coin-flip sequences and pick the one that looks more random.

1. **A pure streak always gets about 10% of choices.** When one sequence is all heads or all tails,
   people pick it about 10% of the time, whatever it is paired with.
   - Three of the four seeds and the blend predict that the streak does worse the more random
     its opponent looks (blend: 18% → 4%). The fourth, `finite_experience_occurrence`, is flat
     but too high, at about 18%.
   - All three winners stay flat at about 10%.
   - Filter: `streak_floor_pair == 1`, binned by the other sequence's number of switches.
2. **Varied run lengths look more random.** With the number of switches held equal, people prefer
   the sequence whose runs vary in length (e.g. runs of 1, 2, 5 over 1, 6, 1).
   - People choose the more varied sequence 63% of the time; the seeds and blend predict 50–52%.
   - Run 1's winner predicts 65%, run 2's 58% and run 3's 56%.
   - Filter: `variety_pair == 1`.
3. **Some people reverse the usual preference.** About 13% of people agree with the majority on
   no more than 55% of trials, and some systematically prefer streaks.
   - Seeds predict the same for everyone, while the winners have person-level parameters.
   - Use `person_agree_majority`, and compare winners' per-person predictions.

### Number game (`ng_trials.csv`)

People see example numbers from a hidden rule (e.g. 16, 8, 2) and say whether a target number
also fits it.

4. **One example extends to its multiples.** After a single example, people say yes to its
   multiples (16 → 64) 63% of the time, against 49% for other targets.
   - Every seed and the blend predict no gap, because the seeds have no "multiples of 16" rule.
   - The winners predict the gap.
   - Filter: `n_examples == 1`, split by `target_multiple_of_single_example`.
5. **A firm no to rule-breakers, a soft yes to members.** People are hedged about targets that
   fit the rule and confident about targets that break it:
   - targets keeping a shared final digit get 82% yes;
   - targets breaking that digit get 11%;
   - targets breaking a shared parity get 16%.

   No seed, seed with a refitted link, or blend gets all three; the winners do. Filters:
   `shared_digit_set` × `target_keeps_digit`, and `shared_parity_set` × `target_keeps_parity`.

### RSA reference games (`rsa_trials.csv`, `rsa_display_classes.csv`)

People see a display of objects and a word, and choose which object the speaker meant.

6. **The candidate in the middle is passed over.** When the word fits objects with different
   numbers of features, people choose the simplest (64%) or the richest (19%) and rarely the one
   in between (15%).
   - All 17 starting models give the middle object at least as much as the richest (24–37%).
   - Chain 0's final predicts 64%, 13% and 22%.
   - Filter: `interior_display == 1` in `rsa_display_classes.csv`, grouped by `role`.

## Files and columns

| file | rows |
|---|---|
| `sr_trials.csv` | 23,104 trials, 359 people, 416 sequence pairs (sr-oct26 runs 1–3 pooled) |
| `ng_trials.csv` | 22,680 trials (experiment-3 responses of the November dataset runs 1–3 pooled) |
| `rsa_trials.csv` | 14,404 trials, 1,108 people (the pre-registered experiment; test trials, no catch trials) |
| `rsa_display_classes.csv` | one row per display × choice option, with every model's full predicted distribution and the observed counts. Use it for choice-share plots. |
| `provenance/` | the scripts that produced these files |

Prediction columns start with `p_`. The rest of the name says which model.

### Model names

| domain | winners | seeds | blend |
|---|---|---|---|
| subjective randomness | `win_run1` = `personal_pattern_detection_gain`, `win_run2` = `edge_streak_primacy_recency_pattern_rule`, `win_run3` = `mild_symmetric_trick_coin_streak_weary` | `seed_local_representativeness`, `seed_motif_stack`, `seed_falk_konold_dp`, `seed_finite_experience_occurrence` | `seed_blend_cv` |
| number game | `win_run1` = `span_fallback_rule_violation`, `win_run2` = `neighbourhood_doubt_three_example_memory`, `win_run3` = `exception_weighted_rule_rejection` | `seed_bp_independent`, `seed_bp_mixture`, `seed_tenenbaum`; `*_relinked` = the same seed with a refitted logistic intercept and slope | `seed_blend_cv` |
| RSA | `final_chain0` (the best final), `final_chain1`, `final_chain2` | 5 textbook `seed_*` models and the 12 `promoted_*` models the chains started from | `seed_blend_cv` |

### `sr_trials.csv`

| column | meaning |
|---|---|
| `person` | integer participant id (as in the committed `pooled_runs123/responses.csv`) |
| `trial_index` | trial number |
| `sequence_a`, `sequence_b` | the two sequences shown |
| `chose_a` | 1 = the person picked `sequence_a` as more random |
| `length` | sequence length |
| `switches_a`, `switches_b` | number of H↔T switches |
| `constant_a`, `constant_b` | 1 = the sequence is all H or all T |
| `run_variety_a`, `run_variety_b` | run-length variety (the run-1 winner's `run_variety` feature) |
| `p_a_*` | each model's P(choose `sequence_a`). The winners' predictions differ by person; the seeds' do not. |
| `streak_floor_pair` | 1 = exactly one sequence is constant, length ≥ 5 (pattern 1) |
| `variety_pair` | 1 = equal switches, different run variety (pattern 2) |
| `person_agree_majority` | this person's share of trials agreeing with the majority choice for the pair (pattern 3) |

### `ng_trials.csv`

| column | meaning |
|---|---|
| `run`, `trial_index` | source run and trial |
| `number_set` | the examples, in the order shown |
| `target` | the number asked about |
| `said_yes` | 1 = yes |
| `n_examples` | number of examples |
| `cv_fold` | which held-out fold produced the predictions (5 folds by example set) |
| `p_yes_*` | each model's P(yes), predicted for held-out example sets |
| `target_is_example` | 1 = the target is one of the examples |
| `target_multiple_of_single_example`, `target_factor_of_single_example` | pattern 4 |
| `shared_digit_set`, `target_keeps_digit` | all examples end in the same digit, and whether the target does too |
| `shared_parity_set`, `target_keeps_parity` | likewise for odd/even. A shared-digit set is also a shared-parity set. |
| `common_divisor_set`, `target_keeps_divisor` | the examples share a divisor ≥ 3, and whether the target is a multiple of it |
| `distance_outside_range` | how far the target lies outside the examples' range (0 = inside) |

### `rsa_trials.csv`

| column | meaning |
|---|---|
| `person` | de-identified pseudonym (as in `data/rsa_shared/`) |
| `condition` | design condition |
| `display` | id of the display, shared with `rsa_display_classes.csv` |
| `objects` | each object's binary feature vector |
| `utterance` | index of the word's feature |
| `word`, `feature_names` | the word and what each feature is |
| `choice` | index of the object chosen |
| `choice_class` | identical objects share a class |
| `chosen_feature_count` | number of features the chosen object has |
| `chosen_role` | `simplest_match` / `middle_match` / `richest_match` among the objects the word fits; also `match_all_same_count`, `no_match` or `no_word` |
| `interior_display` | 1 = the display has a middle candidate (32 displays) |
| `p_chosen_*` | each model's probability of the choice the person made |

### `rsa_display_classes.csv`

| column | meaning |
|---|---|
| `display`, `objects`, `utterance`, `interior_display` | as above |
| `n_trials` | trials on this display |
| `choice_class` | the choice option |
| `member_objects`, `n_members` | the objects in that class |
| `feature_count`, `role` | as above, for this class |
| `observed_count` | how many people chose it |
| `p_*` | each model's predicted probability of choosing this class |

To plot pattern 6, sum `observed_count / n_trials` and each `p_*` within display × role, then
average over displays, weighting by `n_trials`.

## Caveats

- **Subjective randomness:** predictions are posterior means from NUTS fits to these same data, so
  they are in-sample. The blend is cross-validated (5 folds by pair).
- **Number game:** predictions come from MAP fits, cross-validated by example set. Their totals are
  within about 50 nats of the published values for the winners.
- **RSA:** predictions were frozen before the experiment ran, so they are fully out-of-sample.
- **Blends:** the subjective-randomness and number-game blends are logistic stacks of the seeds'
  predictions; the RSA blend is a mixture of the 17 starting models. They can only reweight the
  seeds, not add anything new.

## Provenance

- **Scripts:** `provenance/` holds the analysis scripts as they were run on 2026-11-18.
  `export_phenomena.py` writes these files. The scripts use hard-coded temporary paths, so they
  record what was done rather than being ready to rerun.
- **Ids and sources:**
  - The SR and RSA ids are the de-identified ones already committed elsewhere in the repo.
  - The number-game responses carry no ids. They were drawn from the gitignored
    `data/number-game/` dataset, and the same responses are committed under
    `data/results/number_game_dataset-nov26/`.

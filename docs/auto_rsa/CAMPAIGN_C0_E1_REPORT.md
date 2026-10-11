# Campaign, chain 0, experiment 1: report

From the local Sherlock session, 2026-10-11, for the driver session and the PI.
Handoff: `HANDOFF_live.md` §4.

| | |
|---|---|
| job | 47331282_0: COMPLETED, 12 h 12 min (21:36 – 09:48 PDT) |
| frozen code | **b162c96011677460d3180623153f6f6fc638c78f** (`$WORK_ROOT/code_commit`) |
| work root | `$SCRATCH/auto-psych/rsa_campaign`; cell `live_rsa_c0`; agent tree `63cfc8f3a326e982` |
| Prolific study | `6acb1fc4da7e6ad07dfb3f22`: 200 places, $0.50, listed at 3 min, desktop, earlier-participants blocklist on (it holds the pilot) |

The outputs are in `data/rsa/live/live_rsa_c0/`. They are the aggregates and model files,
brought back without:
- trial-level CSVs;
- `raw_collected/` or `participant_ids.json`;
- agent logs or fit caches.

From `private/`, only `outer_config.json` and `experiment1/prospective.json` came back. The
scan for keys, tokens and 24-character Prolific ids is clean.

**What ran, and the change of plan:**
- The launch was 3 chains × 1 experiment, as §4 says. The PI then clarified the plan: his
  "experiment" is our chain and his "round" is our experiment. So it is chain 0 through
  experiments 1–3 first (~$400), then chains 1 and 2 (~$800) if it works.
- Tasks 1 and 2 were cancelled while still designing, about 35 min in. Neither created a
  Prolific study.
- **Launcher bug, to fix before the next launch:** `scripts/rsa/live/launch.sh` line 30 (the
  "this launch … ~$" print) is a Python SyntaxError. The `\"` inside bash single quotes reaches
  Python literally. Under `set -e` the launcher exits before its prompt whenever
  `N_EXPERIMENTS` is set.
  - The launcher had already printed the config check and the cost ($133 a study with the
    fee) before it failed.
  - I submitted with the launcher's own `_env.sh`, config check and `sbatch` line, by hand
    (`$WORK_ROOT/launch_campaign_e1.sh`), with `CONFIRM_LIVE_RECRUITMENT=1 PROLIFIC_MODE=live
    N_EXPERIMENTS=1`.

## 1. Recruitment and data

**Timeline:**
- design 21:36 – 22:31 (55 min);
- study published 22:33;
- all 200 places filled by about 00:00 (about 1.5 h).

**Prolific** (from its API; no ids printed):

| | n |
|---|---|
| submissions | 211 |
| approved automatically | 176 |
| **held for review** (all `time_taken_under_auto_approval_threshold`, 56–89 s) | **24** |
| returned (place refilled) | 11 |

**Completion time** (the 200 completers): median **144 s (2.4 min)** against the 3-min
listing; p10 86 s, p90 290 s, min 56 s, max 1,479 s. The pilot (12 choices) had a median of
119 s. At the median, 50¢ is about $12.50/h.

**Catch-trial exclusions:** **15 of 200 (7.5%)**, so **185 people are included**
(`data/participants.json`: `max_catch_errors` 0, `without_trials` 0, `other_study` 0). The
pilot had none.
- **The fast people are not the careless ones.** Only 1 of the 24 held submissions failed a
  catch trial; 14 of the 15 excluded were auto-approved.
- The excluded people's times (81–554 s) spread across the range.

**For the PI:**
- approve the 24 held submissions (23 passed both catch trials);
- the dashboard's messages are not readable through the API.

**Cost:** 200 × $0.50 + fee, about **$133** (the launcher's figure).

**The trials:** 2,775 live trials (185 × 15) over 53 displays (52 designed + catch).

## 2. The design

`experiment1/design/eig.json`:
- pool 1,597; 13 models (6 carried + 7 bar); `n_withheld` 0; **`screened_out` empty**.

**Quotas: all met** (count ≥ minimum):

| quota | minimum | count |
|---|---|---|
| objects=2 | 8 | 8 |
| objects=3 | 13 | 13 |
| query=prior | 10 | 16 |
| query=word | 26 | 36 |

**Kinds:** 2×2 word 1; 2×3 prior 1, word 2; 2×4 prior 2, word 2; 3×3 word 2; 3×4 prior 4,
word 7; 4×3 prior 1; 4×4 prior 8, word 22. There were 45 EIG picks, and each display got
50 responses at N=200.

**Power:**

| | joint EIG | p(correct) | mean posterior on truth |
|---|---|---|---|
| with quotas | 3.478 of 3.696 bits | **0.937 ± 0.005** | 0.910 |
| free | 3.485 bits | 0.942 ± 0.005 | 0.912 |

## 3. Prospective score (`private/experiment1/prospective.json`)

This is the models in play before the experiment, scored on the new live data:
- best live model `competitor_confusion_speaker`;
- `live_vs.best_seed` (`seed:rsa_l1`): **+44.8 ± 54.9 nats**, not distinguishable;
- `live_vs.best_promoted`: 0 (it is itself the best promoted model);
- `committed_vs` is null in experiment 1. Claim 2's test starts in experiment 2.

## 4. The inner loop

5 rounds × 6 slots, about 9.8 h (00:00 – 09:47). **The best model changed every round:**

| round | critiqued incumbent | best model after the round |
|---|---|---|
| seed | | `rsa_l2_singleton_feat_color_valence_l0` |
| 1 | `rsa_l2_singleton_feat_color_valence_l0` | `rsa_l2_confusion_singleton_l0` (refine incumbent) |
| 2 | `rsa_l2_confusion_singleton_l0` | `limited_attention_listener` (explore) |
| 3 | `limited_attention_listener` | `expected_success_confusion_attention` (refine chosen) |
| 4 | `expected_success_confusion_attention` | `candidate_load_attention_listener` (refine incumbent) |
| 5 | `candidate_load_attention_listener` | **`candidate_load_codability_listener`** (refine incumbent, repair) |

**The exported best:** `candidate_load_codability_listener`. It is a depth-2 pragmatic
listener with:
- attentional capture that rises with the number of referential candidates;
- a "codability" prior (objects with distinguishing features are likelier referents);
- an expected-success speaker;
- competitor-confusion avoidance;
- a salience prior.

**Against the starting incumbent** (the final standing, grouped CV):
- **on the live trials: +82.8 ± 22.0 nats** (−1,945.9 vs −2,028.7 over 2,775 trials);
- on the existing data by source:
  - mayn_demberg_2022 +101;
  - mayn_demberg_2023 +87;
  - pragmods +60;
  - sikos_2021 −5;
  - mayn_demberg_2026 −8.

So the gain on live data does not cost the literature fit; it improves three of its five
sources.

**Ledger:** 30 admitted, 4 rejected, 26 pruned.
- **Rejected:** 3 as not novel (RMSE 0.0000–0.0010 against the 0.002 threshold), 1 that did
  not converge (R-hat 1.54; repaired).
- **Pruned at the end:**
  - 22 by the guard, more than 4 clustered SE behind on the existing data. This includes
    **every starting model**: e.g. `rsa_l2_singleton_feat_color_valence_l0` 291 ± 44 behind,
    `rsa_l2` 708 ± 121.
  - 4 more than 2 clustered SE behind on the live trials (35–56 nats).

**The exported live set (10):**
- `candidate_load_codability_listener`, `inhibitory_load_listener` (0.4 ± 7.3 behind on
  live), `candidate_load_cost_attention`, `candidate_load_attention_listener`,
  `ambiguity_attention_listener`, `success_confusion_cost_attention`,
  `expected_success_confusion_attention`, `limited_attention_listener_2`,
  `candidate_load_outlier_listener`, `literal_pragmatic_mixture_listener`;
- all within 15 nats of the best on live.

**`carry.json` does not exist yet.** `run.py` writes it at the start of experiment 2, when it
picks at most 8 distinct models from these 10, so it will be in the experiment-2 report.

## 5. The critic, now on the live trials only

`CRITIQUE_CONTEXT.md` confirms the scope: 2,775 trials over 193 displays from
`source == "auto_psych"`.

| round | status | attempts | statistics / evaluated / p ≤ 0.05 / survive FDR | p range |
|---|---|---|---|---|
| 1 | critiqued | 1 | 8 / 8 / 8 / 8 | 0.002 – 0.006 |
| 2 | critiqued | 1 | 8 / 8 / 8 / 8 | 0.002 – 0.020 |
| 3 | critiqued | **2** | 8 / 8 / 8 / 8 | 0.002 – 0.008 |
| 4 | critiqued | 1 | 8 / 8 / 8 / 8 | 0.002 – 0.014 |
| 5 | critiqued | 1 | 8 / 8 / 8 / 8 | 0.002 – 0.008 |

- **Round 3's first attempt** wrote no `test_stats/` at all ("no usable test statistic"). The
  retry (`critique_retry_1/`) wrote 8.
- **Nothing was set aside:** no `broken_statistics/`, and every `error` is null.

**The recurring discrepancies.** Each row is observed vs the incumbent's null, as z per
round:

| statistic | observed | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| `ubiquitous_utterance_max_feature_rate` (feature-rich object when the word fits all) | 0.332 | +7.2 | +4.6 | +7.1 | +4.1 | +3.3 |
| `two_object_scalar_implicature_rate` | 0.630 | −5.4 | −3.3 | −5.7 | −3.7 | −3.5 |
| `utterance_singleton_over_duplicate_rate` | 0.594 | −4.4 | −4.8 | −4.0 | −3.7 | −3.5 |
| `mumble_singleton_choice_rate` | 0.644 | −3.3 | −4.1 | −2.7 | −3.8 | −3.5 |
| foil rates (`four_object_foil_error_rate`, then `single_foil_choice_rate`) | 0.037 / 0.062 | +5.1 | +5.3 | +3.9 | +4.4 | +3.8 |
| `two_object_mumble_feature_rich_rate` | 0.750 | — | +2.4 | +2.9 | +3.5 | +3.6 |
| `three_match_intermediate_feature_rate` | 0.092 | — | — | — | −3.75 | −3.7 |

**Reading:**
- On the live trials alone, the p values are no longer all at the floor, and **the largest
  discrepancy shrank as the incumbent changed** (z +7.2 → +3.3). The critique now tracks the
  loop's progress.
- Still, every statistic is significant in every round. The best model still under-predicts
  feature-rich choices when the word fits every object, and over-predicts the scalar
  implicature on two-object displays.

**A question for the driver: catch trials sit in the critiqued and fitted data.** The 2,775
trials are 185 × 15, so they include the 2 catch trials per person.
- After the exclusion rule (`max_catch_errors` 0), accuracy on them is 1.0 by construction.
- The critic noticed: `catch_trial_literal_accuracy` (r1, r2) and `popout_literal_accuracy`
  (r4, r5) are observed at 1.000 vs a null of 0.97–0.99, z +2.2 to +3.4. They measure the
  exclusion rule, not listeners.
- The same 370 trials, perfect by selection, also enter every model's fit, which pushes the
  lapse rates down.
- Should catch trials be dropped from the modelled data, or kept on purpose?

## 6. Did the candidates use it?

**Delivery:** every candidate dir has `critiques.md`.

**Use:** nearly every hypothesis argues from a discrepancy, **31 of 34 attempts** by my read.
They mostly target the two largest:
- feature-rich choices on ubiquitous-word displays;
- the over-strong scalar implicature (the "moderates extreme scalar exclusion" phrasing
  recurs).

Later rounds also target the foil and pop-out statistics: `contrastive_outlier_listener`,
`feature_binding_listener`, `inhibitory_load_listener`.

**The three exceptions:**
- `rsa_l2_confusion_singleton_l0` (round 1's winner, refine incumbent): it describes the
  added confusion term but no discrepancy.
- `fewest_features_listener` (round 3, explore): a heuristic with no reference to the
  critique.
- **`limited_attention_listener_2` (round 3, refine incumbent): a stale hypothesis that
  passed the new copy check.** Its `hypothesis.md` is a single 20-word sentence, its parent's
  first sentence verbatim. It is not identical to the parent's full hypothesis, so the
  word-for-word rule (#51) let it through.
  - It is in the exported live set.
  - Does the rule need to catch a prefix, or a hypothesis with no stated difference from its
    parent?

## 7. Spend

| | calls | tokens | cost |
|---|---|---|---|
| `rsa:candidate` | 34 | 326.8 M | $44.57 |
| `rsa:critique` | 6 | 48.7 M | $6.78 (13%) |
| **agents total** | 40 | 375.4 M | **$51.35** |
| Prolific (200 × $0.50 + fee) | | | **~$133** |

There were no usage-limit hits.

## 8. Before experiments 2–3 (PI go-ahead)

1. **Fix `launch.sh` line 30.** Otherwise the resubmit has to bypass the launcher again.
2. **Decide on catch trials** in the modelled and critiqued data (§5).
3. **Optionally,** tighten the copy check for a parent-prefix hypothesis (§6).
4. **Then resubmit chain 0 only:**
   `PROLIFIC_MODE=live N_EXPERIMENTS=3 CHAINS_TO_RUN=0 WORK_ROOT=$SCRATCH/auto-psych/rsa_campaign CONFIG=<the copy>`.
   - Experiment 1's stages are kept, and the experiments 2–3 cost about $266.
   - A fix commit changes the code; `stage_code.sh` refuses a work root staged from other
     code. So the driver should say whether the fixed commit replaces the freeze, or only
     the launcher (which is not staged) changes.

# Campaign, chain 0 (experiments 1–3): report

From the local Sherlock session, 2026-10-13, for the driver session and the PI. Experiment 1 is
reported in detail in `CAMPAIGN_C0_E1_REPORT.md`; this report adds experiments 2 and 3 and the
chain as a whole.

**The runs:**

| | |
|---|---|
| code | experiment 1 on **b162c960**; experiments 2–3 on **e5fb029d** (`code_history`: one recorded re-stage, "campaign log #54-57") |
| jobs | 47331282_0 (experiment 1) · 47396066_0 (experiment 2; failed at experiment 3's publish, Prolific funds) · 47514554_0 (experiment 3, COMPLETED 04:47 on 10-13) |
| Prolific | `6acb1fc4…` (experiment 1), `6acbe441…` (experiment 2), `6accf48f…` (experiment 3) |

The outputs are in `data/rsa/live/live_rsa_c0/`. They are brought back as before:
- no trial-level CSVs, `raw_collected/`, `participant_ids.json`, agent logs or fit caches;
- from `private/`, only `outer_config.json` and `experiment*/prospective.json`.

The scan for keys, tokens and 24-character Prolific ids is clean.

## 1. Claim 2: the committed model predicts the next experiment

| experiment | committed model (the previous experiment's export) | vs best seed | vs best promoted seed |
|---|---|---|---|
| 2 (2,509 trials) | `candidate_load_codability_listener` | **+126.9 ± 37.4** (3.4 SE) | +32.2 ± 23.3 (1.4 SE) |
| 3 (2,405 trials) | `prototype_typicality_bounded_listener` | **+120.2 ± 28.0** (4.3 SE) | **+43.6 ± 14.4** (3.0 SE) |

These are differences in log predictive density on the new experiment's data, before anything is
refitted to it (`prospective.json` `committed_vs`).
- The best seed is `seed:rsa_l1_salience` in both experiments.
- The best promoted seed is `promoted:rsa_l2_singleton_feat_color_familiar_l0` in both.

The committed model beats all 17 bar models in both experiments (`each_bar_model`). The smallest
margin is 1.4 SE in experiment 2 and **3.0 SE** in experiment 3.

## 2. Recruitment and data

| experiment | recruited | kept (catch) | median time | p10–p90 | held for speed | returned |
|---|---|---|---|---|---|---|
| 1 (15 choices, catch modelled) | 200 | 185 (15 excluded) | 144 s | 86–290 s | 24 | 11 |
| 2 | 201 | 193 (8) | 124 s | 77–200 s | 39 | 6 |
| 3 | 201 | 185 (16) | 135 s | 85–248 s | — (all 200 now approved) | 7 |

- Each study filled in about 30–90 min.
- Each cost about $133 (200 × $0.50 + fee): about **$400** for the chain.
- **Experiment 3's first publish failed:** the account held $125.33 against $133.33. The PI
  topped up, the draft was published by hand, and the run resumed (`CAMPAIGN_LOG.md`).

## 3. The designs

| experiment | models (carried + bar) | power (quotas) | free | quotas | screened out |
|---|---|---|---|---|---|
| 1 | 13 (6 + 7) | 0.937 | 0.942 | all met | none |
| 2 | 21 (8 + 13) | 0.753 | 0.757 | all met | none |
| 3 | 20 (8 + 12) | 0.705 | 0.715 | all met | none |

- Power falls as the carried models become close relatives of each other.
- **The carried set:**
  - experiment 2 kept 8 of 10 (2 over the cap);
  - experiment 3 kept 8 of 12 (4 over the cap, RMSE 0.003–0.013 to a kept model);
  - none were merged.

## 4. The inner loops

**The best model after each round** (round 0 is the seed step):

| experiment | rounds 0 → 5 |
|---|---|
| 1 | `rsa_l2_singleton_feat_color_valence_l0` → `rsa_l2_confusion_singleton_l0` → `limited_attention_listener` → `expected_success_confusion_attention` → `candidate_load_attention_listener` → **`candidate_load_codability_listener`** |
| 2 | `inhibitory_load_listener` → `inhibitory_codability_listener` → `prototype_typicality_listener` → **`prototype_typicality_bounded_listener`** (rounds 3–5 unchanged) |
| 3 | `bounded_typicality_outlier_listener` → `avoidable_ambiguity_speaker` → `focal_bounded_info_gain_listener` → `boundary_polarization_listener` → **`boundary_avoidable_ambiguity_listener`** |

**Gain over the committed model** (grouped CV on the cumulative live trials):
- **Experiment 2: +4.3 nats.** The committed `candidate_load_codability_listener` was then
  pruned by the guard, 62 nats behind on the existing data.
- **Experiment 3: +47.5 ± 13.4 nats** for `boundary_avoidable_ambiguity_listener` over
  `prototype_typicality_bounded_listener`.

**Ledger:**

| experiment | admitted | rejected | pruned / retired |
|---|---|---|---|
| 1 | 30 | 4 | 26 |
| 2 | 27 | 9 (all not novel) | 20 / 3 |
| 3 | 25 | **15** | 28 |

Experiment 3's 15 rejections:
- 8 not novel;
- **5 hit the 30-min fit limit**;
- 2 did not converge.

The data grow with each experiment, and the larger candidates now run into the fit time limit.

**The final export** (experiment 3) holds 5 live models. The best,
**`boundary_avoidable_ambiguity_listener`** (17 parameters), adds two mechanisms to the attention
models:
- **Boundary polarization:** among several matching objects, attention anchors on the extremes
  (the fewest-feature target and the most-feature "prototype") and suppresses objects in the
  middle of the range.
- **Avoidable ambiguity:** the speaker is penalized for an ambiguous word only when the intended
  object had a word that would have picked it out uniquely. So a competitor with a unique
  alternative word is ruled out under a shared word, while targets without one are favored.

## 5. The critic

**Significant statistics per round** (of 8, on the live trials):

| experiment | per round |
|---|---|
| 1 | 8, 8, 8, 8, 8 (round 3 needed its retry) |
| 2 | 8, 8, **6**, 8, **7** |
| 3 | 8, 8, 8, 8, 8 |

**The last critique (experiment 3, round 5, incumbent `boundary_polarization_listener`):** its
largest misses are about duplicates and the object prior.

| statistic | observed | predicted |
|---|---|---|
| three-object target vs a duplicated competitor | 0.68 | 0.81 |
| mumble trials, duplicated all-feature object | 0.59 | 0.48 |
| mumble trials, singleton | 0.61 | 0.68 |
| four-object multi-feature target | 0.77 | 0.64 |

The same misses run through every round since experiment 2 (`CAMPAIGN_C0_E1_REPORT.md` §5 and
the summary sent to the PI on 10-12). The scalar-implicature misses of experiment 1 have largely
gone.

**A minor quirk for the driver:** in experiment 3, the critic's 5-s statistic timeout (SIGALRM,
`src/critique/ppc.py` `_handler`) once fired inside a garbage-collector callback. Python printed
"Exception ignored … TimeoutError: test statistic exceeded 5s" and carried on, and the critique
completed (8 of 8). A timeout raised there is discarded, so that statistic's call was not cut
short at that moment.

## 6. Time and spend

| experiment | inner loop | agent spend (candidate / critique) |
|---|---|---|
| 1 | 9.8 h | $51.35 ($44.57 / $6.78) |
| 2 | 16.8 h (rounds 1.8–4.2 h) | $58.60 ($52.63 / $5.97) |
| 3 | 19.4 h | $66.88 ($58.07 / $8.81) |
| **chain 0** | | **$176.83** agents + **~$400** Prolific |

The loops grow with the data. That is why chains 1 and 2 need about 55 h for three experiments,
more than `outer_live.sbatch`'s 48 h (`CAMPAIGN_LOG.md`, 10-12).

## 7. The other chains (as of 10-13 05:00)

- **Chain 1:** experiment 1 done (best `softmax_belief_listener`). Experiment 2's study
  `6acdfe10…` filled overnight, and its inner loop has run since 04:12. A continuation
  (47575694_1, 7-day limit) is queued for when the current job reaches its 48 h limit.
- **Chain 2:** experiment 1 done (best `bounded_capacity_redundancy_l2`; its study was paused by
  Prolific at 196/200 and the 196 were modelled). It is **held**: its Hosting site is still
  flagged by Safe Browsing (`SAFE_BROWSING_C2.md`).

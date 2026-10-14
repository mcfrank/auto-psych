# Campaign, chain 1 (experiments 1–3): report

From the local Sherlock session, 2026-10-14, for the driver session and the PI. The format is
as for chain 0 (`CAMPAIGN_C0_REPORT.md`).

**The run:**

| | |
|---|---|
| code | e5fb029d throughout (no re-stage) |
| job | 47518520_1: **COMPLETED** 10-14 01:47 in 40 h 11 min, inside its 48 h limit. The queued continuation 47575694_1 was never needed; Slurm cancelled it when its `afternotok` dependency could no longer be met. |
| Prolific | `6acd1607…` (experiment 1), `6acdfe10…` (experiment 2), experiment 3 published 10-13 12:40 |

The outputs are in `data/rsa/live/live_rsa_c1/`, as before:
- no trial-level CSVs, `raw_collected/`, `participant_ids.json`, agent logs or fit caches;
- from `private/`, only `outer_config.json` and `experiment*/prospective.json`.

The scan is clean.

## 1. Claim 2

| experiment | committed model | vs best seed | vs best promoted seed | smallest margin over the 17 bar models |
|---|---|---|---|---|
| 2 (2,145 trials) | `softmax_belief_listener` | **+165.4 ± 43.7** (3.8 SE) | **+32.4 ± 12.0** (2.7 SE) | 2.7 SE |
| 3 (2,509 trials) | `contrastive_extremity_noisy_speaker` | **+139.3 ± 34.8** (4.0 SE) | **+55.9 ± 17.9** (3.1 SE) | 3.0 SE |

**Chain 1 is the cleanest claim-2 result so far.** In both experiments, the committed model beats
every bar model by at least 2.7 SE on data it had not seen.

## 2. Recruitment and data

| experiment | recruited | kept (catch) | median time | p10–p90 | returned |
|---|---|---|---|---|---|
| 1 | 200 | 189 (11 excluded) | 119 s | 79–212 s | 8 |
| **2** | 201 | **165 (36 excluded, 18%)** | **162 s** | 96–340 s | **14** |
| 3 | 200 | 193 (7) | 126 s | 82–242 s | 4 |

**Experiment 2 is an outlier:**
- 18% excluded on catch trials (5–8% in every other study);
- slower, and more returns.

It recruited between 02:46 and 04:07 PDT. That is UK morning, so a different part of the pool
than the daytime studies. This is worth a look before the paper. The exclusion rule worked as
designed; the question is whether the night-time pool differs.

Each study cost about $133.

## 3. The designs

| experiment | models | power (quotas) | free | quotas | screened out | carried |
|---|---|---|---|---|---|---|
| 1 | 13 | 0.949 | 0.950 | met | none | — |
| 2 | 19 | 0.948 | 0.945 | met | none | 8 of 12 (4 over cap) |
| 3 | 20 | 0.915 | 0.915 | met | none | 8 of 12 (4 over cap) |

Chain 1's designs keep their power across experiments, unlike chain 0's (0.94 → 0.75 → 0.71).

## 4. The inner loops

**The best model after each round** (round 0 is the seed step):

| experiment | rounds | best |
|---|---|---|
| 1 (13.0 h) | `rsa_l2_singleton_feat_color_familiar_l0` → `rsa_l1_singleton_feat_color_familiar` → `attentive_display_listener` → **`softmax_belief_listener`** (rounds 3–5 unchanged) | |
| 2 (6.8 h) | **`contrastive_extremity_noisy_speaker`** from round 1; **stopped after round 3** (two stale rounds) | +29.1 ± 19.2 over the committed model |
| 3 (12.4 h) | `contrastive_extremity_noisy_speaker` → `contrastive_graded_noisy_speaker` → `feature_aspect_qud_listener` → **`feature_surprisal_ensemble_listener`** | +19.2 ± 16.7 over the committed model |

**Rejections:**
- experiment 1: 8 (6 not novel, 2 unconverged);
- experiment 2: 10 (**8 unconverged**, 2 not novel);
- experiment 3: 12 (7 not novel, 4 unconverged, 1 non-finite ELPD).

Unlike chain 0, no candidate hit the 30-min fit limit.

**The final model: `feature_surprisal_ensemble_listener`.** It has **9 parameters**, about half
chain 0's final (17). Its hypothesis, in short:
- Attention and the speaker's expected topic follow each object's **feature surprisal** in context.
- **Identical objects form an ensemble** that shares attention equally.
- The featureless object acts as the scene's contrastive ground, and the most elaborated object
  anchors prominence. This compresses intermediate objects and keeps empty singletons from
  dominating feature-rich duplicates.

It targets the same duplicate and singleton misses the critic reported in chain 0.

**The critic:** 8 of 8 significant in every round of experiments 2 and 3; experiment 1's first
round had 5 of 8.

## 5. Spend

| experiment | agents (candidate / critique) |
|---|---|
| 1 | $53.18 ($47.13 / $6.05) |
| 2 | $31.08 ($27.60 / $3.48), the short loop |
| 3 | $55.90 ($50.26 / $5.64) |
| **chain 1** | **$140.16** agents + **~$400** Prolific |

## 6. Across chains

The pooled fit (`data/rsa/live/finals_pool_preds.json`, 10-13) used chain 1's experiment-2
export, `contrastive_extremity_noisy_speaker`, because experiment 3 had not finished. Rerunning it
with `feature_surprisal_ensemble_listener`, and chain 2's final once it exists, takes about
10 minutes.

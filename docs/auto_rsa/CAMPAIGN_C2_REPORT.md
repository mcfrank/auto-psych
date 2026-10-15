# Campaign, chain 2 (experiments 1–3), and the campaign across chains: report

From the local Sherlock session, 2026-10-14, for the driver session and the PI. The format is
as for chains 0 and 1 (`CAMPAIGN_C0_REPORT.md`, `CAMPAIGN_C1_REPORT.md`).

**The runs** (all on code e5fb029d):

| | |
|---|---|
| jobs | 47523801_2 (experiment 1; cancelled in round 2, 10-12) · 47575675_2 (experiment 1 only; held for the flagged site) · 47676907_2 (experiments 2–3, 7-day limit, COMPLETED 10-14 19:02) |
| Prolific | `6acd20b6…` (experiment 1; **paused by Prolific at 196/200**, `SAFE_BROWSING_C2.md`), `6ace99db…` (experiment 2), `6acf8b59…` (experiment 3) |

The outputs are in `data/rsa/live/live_rsa_c2/`, brought back as before. The scan is clean.

## 1. Claim 2

| experiment | committed model | vs best seed | vs best promoted seed | smallest margin over the 17 bar models |
|---|---|---|---|---|
| 2 (2,353 trials) | `bounded_endpoint_prev_redundancy_l2` | **+139.0 ± 51.3** (2.7 SE) | +13.6 ± 27.2 (0.5 SE) | 0.5 SE |
| 3 (2,470 trials) | `competition_endpoint_binding_l2` | **+90.3 ± 34.5** (2.6 SE) | +19.2 ± 15.1 (1.3 SE) | 1.3 SE |

- Chain 2's committed models beat the seeds clearly, as in the other chains.
- But they are **not distinguishable from the best promoted seed**, in either experiment.
- In both experiments, the closest bar model is a promoted `rsa_l2_singleton_…_l0` model.

## 2. Recruitment and data

| experiment | recruited | kept (catch) | median time | p10–p90 | held for speed | returned |
|---|---|---|---|---|---|---|
| 1 | 196 (study paused by Prolific) | 178 (18 excluded) | 115 s | 74–230 s | 42 | 3 |
| 2 | 200 | 181 (19) | 129 s | 83–254 s | 30 | 8 |
| 3 | 200 | 190 (10) | 104 s | 69–198 s | **54, awaiting the PI's review** | 8 |

Each study cost about $133.

## 3. The designs

| experiment | models | power (quotas) | free | quotas | screened out | carried |
|---|---|---|---|---|---|---|
| 1 | 13 | 0.948 | 0.950 | met | none | — |
| 2 | 20 | 0.721 | 0.724 | met | none | 8 of 12 (4 over cap) |
| 3 | 20 | 0.767 | 0.782 | met | none | 8 of 12 (4 over cap) |

## 4. The inner loops

**The best model after each round** (round 0 is the seed step):

| experiment | rounds | best |
|---|---|---|
| 1 (11.2 h, with the restart) | `rsa_l2_singleton_distinct_feat_color_l0` → `feature_redundancy_discounted_l2` → `bounded_capacity_listener_l2` → `bounded_capacity_redundancy_l2` → **`bounded_endpoint_prev_redundancy_l2`** | |
| 2 (14.0 h) | `bounded_endpoint_prev_redundancy_l2` → `salience_lapse_endpoint_prev_l2` → `feature_uncertainty_salience_lapse_l2` → `feature_binding_uncertainty_lapse_l2` → `endpoint_binding_uncertainty_lapse_l2` → **`competition_endpoint_binding_l2`** | (the committed model was not in the final standing) |
| 3 (11.3 h) | `feature_binding_salience_lapse_l2` → `endpoint_binding_salience_lapse_l2` → **`competition_endpoint_binding_lapse_l2`** (from round 2; stopped after round 4) | +2.4 ± 2.1 over the committed model |

**Rejections:**
- experiment 1: 11 (10 not novel);
- experiment 2: 14 (9 not novel, 4 unconverged, 1 non-finite);
- experiment 3: 7.

No candidate hit the fit time limit.

**The final model: `competition_endpoint_binding_lapse_l2`** (13 parameters). It is a depth-2 listener
with:
- contextual endpoint salience (the extremes of the feature range stand out);
- perceptual feature binding;
- lapses captured by salience;
- **lexical competition in the speaker**: labels compete for retrieval, which lowers the speaker's
  precision for objects with more features. That softens the pragmatic penalty against
  multi-feature competitors.

**The critic:** 8 of 8 significant in every round of all three experiments. Experiment 3's first
round needed the retry.

## 5. Spend

| experiment | agents (candidate / critique) |
|---|---|
| 1 | $67.31 ($61.62 / $5.69) |
| 2 | $63.40 ($56.85 / $6.55) |
| 3 | $49.44 ($44.53 / $4.91) |
| **chain 2** | **$180.15** agents + **~$400** Prolific |

## 6. The campaign across chains

**Claim 2:** the committed model on the next experiment's new data, before any refit.

| chain | experiment | vs best seed | vs best promoted seed |
|---|---|---|---|
| 0 | 2 | +126.9 ± 37.4 (3.4) | +32.2 ± 23.3 (1.4) |
| 0 | 3 | +120.2 ± 28.0 (4.3) | **+43.6 ± 14.4 (3.0)** |
| 1 | 2 | +165.4 ± 43.7 (3.8) | **+32.4 ± 12.0 (2.7)** |
| 1 | 3 | +139.3 ± 34.8 (4.0) | **+55.9 ± 17.9 (3.1)** |
| 2 | 2 | +139.0 ± 51.3 (2.7) | +13.6 ± 27.2 (0.5) |
| 2 | 3 | +90.3 ± 34.5 (2.6) | +19.2 ± 15.1 (1.3) |

(SE multiples in parentheses.)
- **Against the best seed:** positive in all 6 tests, each by at least 2.6 SE.
- **Against the best promoted seed:** positive in all 6, and clear of 2 SE in 3 (chain 0's
  experiment 3 and both of chain 1's).

**The final models:**

| chain | final model | parameters |
|---|---|---|
| 0 | `boundary_avoidable_ambiguity_listener` | 17 |
| 1 | `feature_surprisal_ensemble_listener` | 9 |
| 2 | `competition_endpoint_binding_lapse_l2` | 13 |

The three are distinct, as their pooled-fit predictions differ.

**The whole campaign:**
- **Data:** 9 live studies of 200 people (one stopped at 196). 1,659 people kept after catch
  exclusions, 21,567 modelled trials.
- **Spend:** about **$1,200 on Prolific** and **$497 on agents** ($176.83 + $140.16 + $180.15).

**For the paper, to note:**
- chain 1's experiment-2 exclusion rate (18%, a night-time study);
- chain 2's experiment-1 study, paused by Prolific after a false Safe Browsing flag on its site;
- chain 0's experiment 1, which ran on the pre-fix code (catch trials modelled).

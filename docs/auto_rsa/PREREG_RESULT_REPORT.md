# The pre-registered final experiment: result report

From the local Sherlock session, 2026-10-15, for the driver session and the PI. The plan is
`PREREG_FINAL.md`, registered as OSF https://osf.io/s48jv (tag `rsa-prereg-final` = dbb38535).
The handoff is `HANDOFF_prereg.md`.

The records are in `data/rsa/prereg/run/`:
- `prereg_analysis.json`;
- `participants.json` (counts, and the excluded people by run id: no Prolific ids);
- `log_summary.txt` (the job's stage lines).

Nothing else came back. Trial data, `raw_collected/`, `participant_ids.json` and `private/` stay
on Sherlock (`$SCRATCH/auto-psych/rsa_prereg/cells/prereg/`). The scan for keys and 24-character
ids is clean.

## 1. The run

| | |
|---|---|
| live job | 48021934 (`prereg_live.sbatch`), 15:57–18:18 PDT, COMPLETED |
| code | **re-staged 38b585df → 3b931f93** with the reason "prereg freeze and OSF records: docs and config only (campaign log #64-66)". The code is identical to the tag; the change is docs plus `osf_registration`. |
| config | a copy of `rsa_prereg.yaml` outside the checkout, with `confirm_live_recruitment: true` (never committed), plus `CONFIRM_LIVE_RECRUITMENT=1` and the typed "yes" |
| design | `bc7d60e5b84f…` (as frozen): 312 displays, 156 EIG + 156 random |
| predictions | `finals_pool_preds_final.json` `6a0a476f…` and `bar_pool_preds.json` `70457537…`. The sha256s in `prereg_analysis.json` match the config. |
| Prolific | study `6ad15ad5…`, published about 16:00: 1,224 places, $0.50, 3 min (14 max), desktop, country, language and approval ≥ 98% |
| blocklist | **10 studies**: every earlier RSA study (the pilot and all 9 campaign studies). Checked against the account's list. |
| site | `auto-psych-2c5da-prereg-c0.web.app/e1-prereg/`: page 200; Google Safe Browsing "No available data", so not flagged, before and after launch |
| balance | $894.66 available against $813.96 needed |

**Before the live run:**
- **Test deploy:** job 48017994 on 38b585df made draft `6ad1539d…`, never published. It can be
  deleted in the dashboard. A test draft has no blocklist by design (`create_draft_study` adds it
  only in live mode).
- **Preparation:**
  - **The bar fit** (job 47960098, code 88fa9360) converged for all 17 models. It ran first on
    owners, was preempted, and was requeued onto mcfrank.
  - **The design** (job 47967799, code 507b5b9b) took 2 h 13 min on mcfrank.
  - **A staging bug** in `prereg_prepare.sbatch` (job 47895766) was fixed in 88fa9360, with a
    test assertion.

## 2. Recruitment and data

| | |
|---|---|
| submissions | 1,260: 1,025 approved, **199 awaiting review** (198 under Prolific's speed threshold) at 18:50, 28 returned, 8 timed out |
| completion time | median **127 s**; p10 82, p90 238; min 36, max 971 |
| recruited / kept | **1,225 / 1,108** (117 excluded on catch trials, 9.6%; `max_catch_errors` 0); `without_trials` 0, `other_study` 0 |
| modelled | **14,404 trials** over the 312 displays: EIG half 7,209, random half 7,195 |

The run took 1,225 responses for 1,224 places, as the campaign's studies sometimes did with 201.

## 3. The primary test

Each frozen final model is compared with the best promoted seed,
`promoted:rsa_l2_singleton_distinct_feat_color_l0`. The comparison is on predictions frozen
before any of these data existed, one-sided, at α = 0.05 with Holm's correction.

| final (chain) | parameters | difference in lpd | z | p (one-sided) | Holm rejects |
|---|---|---|---|---|---|
| `boundary_avoidable_ambiguity_listener` (0) | 17 | **+197.9 ± 35.1** | 5.64 | 8.5e-9 | **yes** |
| `competition_endpoint_binding_lapse_l2` (2) | 13 | **+60.8 ± 22.9** | 2.66 | 0.004 | **yes** |
| `feature_surprisal_ensemble_listener` (1) | 9 | +8.1 ± 24.1 | 0.34 | 0.37 | no |

**`all_finals_beat_the_bar`: false.** Two of the three finals beat the strongest pre-existing
model on new people; chain 1's final ties it.

## 4. Secondary results

**Against the best starting seed** (`seed:rsa_l1_shared_prior`), all three finals win clearly:

| final | difference | z |
|---|---|---|
| chain 0 | +849.7 ± 82.2 | 10.3 |
| chain 1 | +659.8 ± 82.4 | 8.0 |
| chain 2 | +712.5 ± 79.0 | 9.0 |

**Against each of the 17 bar models** (`each_bar_model`): every difference is positive for every
final. The smallest z over the 17 bars is:
- chain 0: **5.64**;
- chain 2: 2.66;
- chain 1: 0.34.

In each case the smallest margin is against the best promoted seed or one of its two near-twins,
which the design merged (`PREREG_FINAL.md` §4.1). The analysis scores all 17.

**By half of the design:**

| final | EIG half (156 displays, 7,209 trials) | random half (156 displays, 7,195 trials) |
|---|---|---|
| chain 0 | +142.2 ± 26.7 (z 5.33, Holm yes) | +55.7 ± 22.3 (z 2.50, Holm yes) |
| chain 2 | +28.8 ± 19.8 (z 1.46, no) | +31.9 ± 11.5 (z 2.76, Holm yes) |
| chain 1 | +26.5 ± 21.0 (z 1.26, no) | −18.4 ± 11.4 (z −1.61, no) |

- Chain 0's advantage holds on displays chosen without regard to the models.
- Chain 1's final is ahead on the EIG displays and behind on the random ones.

**Ranking by total lpd** (the top of `lpd`):

| rank | model | lpd |
|---|---|---|
| 1 | chain 0 final | −9,949.6 |
| 2 | chain 2 final | −10,086.8 |
| 3 | chain 1 final | −10,139.4 |
| 4–6 | the three `rsa_l2_singleton_…_l0` promoted seeds | −10,147.5 to −10,148.5 |
| 7 | `softmax_surprisal_isolated_costly_l4` | −10,237.1 |
| … | | |
| last | `literal_listener`, `lexical_preemption_heuristic` | about −11,400 |

## 5. Open items (PI)

- Approve the **199 submissions awaiting review** in Prolific (as of 18:50).
- Delete the unpublished test draft `6ad1539d…` (optional).
- Decide on cleaning up Sherlock scratch, after the data-sharing decision:
  - the campaign (`rsa_campaign`) and its agent trees;
  - `rsa_prereg`;
  - the older work roots.

  The trial data and Prolific ids exist only in the `cells/*/private/` and run directories there.

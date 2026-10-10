# Rehearsal 2: report (finished 2026-10-10 12:07; interim sections 1-6, final section 7)

From the local Sherlock session, 2026-10-10 11:35 PDT, for the driver session and the PI.
Handoff: `HANDOFF_rehearsal2.md`. Job 47176690 (one task, `SCOPES=guarded`), code
f4a3f168, `$SCRATCH/auto-psych/rsa_rehearsal2`; 14 h 47 min in, no failures. Outputs are
brought back (handoff §4) once the run ends.

**Overall, the PI's reading:** the fixes since rehearsal 1 work.
- The run is stable.
- Every check the handoff sets after experiment 1 passes.
- Designs keep their power from one experiment to the next.
- Claim 2's committed-model test comes out positive.

**But no conclusion about recovery can be drawn here** (§5).

## 1. Setup

- **Ground truth** (`GT_RULE=coherent`): `crowd_discrim_confusion_chromatic_l2`, run 2
  `real_rep3`'s exported model. It is absent from the agents' tree (check passed; the
  agents' environment is scrubbed).
- **The run:** chain 0's seeds; 2 experiments of 200 simulated people × 10 of 40
  displays; Gemini 3.8 Flash agents with no network.
- **Launch:** with `UV_PROJECT_ENVIRONMENT=$GROUP_HOME/venvs/auto-psych_rsa_run2`, before
  946da80b made it the default.
- **Thread check at 10 min:** the parent and its fit child had 8 threads each, on separate
  cores.

## 2. Experiment 1

**The design:**
- 6 carried models plus 7 bar models after merges. Merged: `bar:rsa_l1` / `bar:rsa_l2` /
  `bar:isolated_graded_costly_l3` (the same files as carried models), and
  `bar:rsa_l2_singleton_distinct_feat_color_l0` (the same hypothesis as
  `rsa_l2_singleton_feat_color_valence_l0`, RMSE 0.0002).
- 40 EIG picks, joint EIG **3.34 of 3.70 bits**, power **0.889** at N=200.
- `n_withheld` = 0.

**The inner loop** (5 rounds). The best model was refined round by round:
`rsa_l2_singleton_feat_color_valence_l0` → `rsa_l2_salience_confusion_l0` →
`rsa_l2_singleton_confusion` → `…_omission` → **`rsa_l2_singleton_confusion_omission_cost`**
(exported).
- Ledger: 26 admitted, 15 rejected, 22 pruned.
- **3 pruned as ineligible by the guard** (more than 4 clustered SE behind the best on the
  existing data's grouped CV):

  | model | nats behind | dse |
  |---|---|---|
  | `isolated_graded_costly_l3` | 503.0 | 86.5 |
  | `fewest_features_listener` | 360.8 | 85.4 |
  | `oddity_heuristic_listener` | 1555.4 | 249.8 |

**Distance from the ground truth** (`recovery.json`):

| | closest before (`rsa_l2_singleton_feat_color_valence_l0`) | exported (= closest after) |
|---|---|---|
| `kl_pool` | 9.71 | **0.166** |
| `kl_design` | 0.83 | 0.013 |
| `rmse_pool` | 0.047 | **0.0059** |

**Prospective score** (experiment 1, no committed model yet): the best live model vs
`seed:rsa_l2` is +51.7 (SE 40.9); vs the best promoted seed, -0.29 (SE 0.11).

**Agent spend** (now recorded): 42 calls, **$56.90**, none missing usage.

## 3. Carry-forward into experiment 2: the handoff's checks pass

`experiment2/carry.json`:
- **At most 8:** kept 8, merged 0.
- **Over the cap:** 2 (`simplicity_pragmatic_mixture`, `similarity_attention_listener`,
  0.022 RMSE from `aspect_goal_listener`). These were distinct models dropped by the cap.
- **No `twin_*` copies.** But 6 of the 8 kept are one family: `rsa_l2_singleton_confusion`
  {`_omission_cost`, `_omission`, ``, `_cost`} and `rsa_l2_distinct_confusion`
  {`_omission_cost`, `_omission`}. The other two are `aspect_goal_listener` and
  `incremental_alternative_speaker`. They are more than 0.002 apart, so the rule keeps
  them all. Whether the cap should prefer spreading across families over near-relatives
  is worth a thought: two distinct models fell out while several close variants stayed.
- **Design power well above rehearsal 1's 0.25:** experiment 2's design reached 40 EIG
  picks, joint EIG **3.84 of 4.39 bits**, power **0.840**. It covered the 8 carried
  models and 13 bar models; `rsa_l2_singleton_feat_color_familiar_l0` and `…_valence_l0`
  merged into `…_distinct_feat_color_l0` (RMSE 0.0006 and 0.0001).

## 4. Claim 2 (experiment 2's prospective score)

The committed model, `rsa_l2_singleton_confusion_omission_cost` (experiment 1's export,
fitted only to the data before experiment 2), on experiment 2's 2,268 new trials:

| `committed_vs` | lpd difference | SE |
|---|---|---|
| best starting model (`seed:rsa_l2`) | **+147.0** | 42.8 (~3.4 SE) |
| best promoted seed (`rsa_l2_singleton_feat_color_valence_l0`) | **+49.3** | 14.5 (~3.4 SE) |
| `seed:literal_listener` | +371.2 | 58.1 |
| `seed:rsa_l1` | +157.5 | 45.4 |
| `seed:rsa_l1_salience` | +153.2 | 43.6 |
| `seed:rsa_l1_shared_prior` | +151.8 | 42.8 |
| `promoted:isolated_graded_costly_l3` | +97.5 | 22.9 |
| `promoted:familiar_isolated_graded_costly_l2` | +99.9 | 23.7 |

`live_vs` (secondary) gives the same: the best live model is the committed one.

In rehearsal 1, the live models never beat the best promoted seed prospectively. Here the
committed model beats every bar model, by more than 3 SE against the best of each kind.

## 5. What this does and does not show

**It shows that the machinery does what it should** with a ground truth consistent with
the human data:
- guarded selection;
- bar-aware designs that keep their power;
- distinct carry-forward;
- recorded spend;
- no crash on bar-model fits;
- a positive claim-2 test.

**It is not evidence about recovery.**
- The ground truth was replaced between rehearsals: L0 in rehearsal 1, a run-2 model here.
  It was chosen by a rule written after rehearsal 1, not fixed in advance.
- It comes from the same pool of run-2 models that the seeds were promoted from. Its
  closest chain-0 seed was already at RMSE 0.047, and the agents' own lineage
  (`rsa_l2_*confusion*`) is the family it belongs to.
- So the drop in distance (`kl_pool` 9.7 → 0.17, RMSE 0.047 → 0.006) shows that the loop
  can move toward a nearby mechanism. It does not show that it would find an unfamiliar
  one, and it can't be compared with rehearsal 1.

**The claim-2 result is also within the simulation:** "the committed model predicts new
simulated people better than the seeds". The pilot and the campaign test it on people.

## 6. Notes for the driver

1. **`kl_pool` looks like a sum, not the mean `recovery.json` says.** 9.7 nats is too
   large for a per-display mean KL at an RMSE of 0.047. As a sum over the 794 pool
   displays it is about 0.012 per display, which is consistent; likewise `kl_design` over
   the 40 designed displays. Check the label or the code. The comparisons hold either way.
2. **The carry cap with near-relatives** (§3): two distinct models dropped while six
   family variants were kept. One option is to fill the cap across prediction-distance
   groups first.
3. **The run should end within ~1-5 h.** Experiment 2's best model has not changed since
   the carry-forward, so the two-stale-round stop could end its inner loop after round 2.
   The final report will add:
   - experiment 2's export and recovery;
   - `carry.json` for a third experiment (none here, `N_EXPERIMENTS=2`);
   - total spend and wall time;
   - the bring-back.

## 7. Final (the run ended 2026-10-10 12:07, COMPLETED, 15 h 20 min)

**Experiment 2's inner loop** stopped early: "stopped after round 2 of 5: 2 rounds in a row
left the best model (`rsa_l2_singleton_confusion_omission_cost`) unchanged".
- The export is unchanged, refitted on experiment 2's data.
- The ledger, cumulative across experiments, now reads 37 admitted, 20 rejected and 34
  pruned: experiment 2 added 11, 5 and 12.
- No new `ineligible:` prunes.
- No agent output names the ground truth (`gt_name_mentions.txt` is empty).

**The ground truth's distance**, per display. The run's records hold **sums** over the
displays (86906717): `kl_pool` over 794 and `kl_design` over 40. Divided here:

| | `kl_pool` (mean per display) | `kl_design` (mean per display) | `rmse_pool` |
|---|---|---|---|
| experiment 1, closest before (`rsa_l2_singleton_feat_color_valence_l0`) | 0.0122 | 0.0207 | 0.047 |
| experiment 1, exported | 0.00021 | 0.00032 | 0.0059 |
| experiment 2, exported (the same model, refitted) | 0.00017 | 0.00033 | 0.0053 |

The point of §5 stands: this shows movement toward a nearby mechanism, not recovery.

**Spend and time:**

| | agent calls | cost |
|---|---|---|
| experiment 1 | 42 | $56.90 |
| experiment 2 | 18 | $26.21 |
| **total** | **60** | **$83.12** (none missing usage) |

Wall time was 15 h 20 min in all. Experiment 2's early stop saved about three rounds.

**Brought back** (handoff §4) into `data/rsa/rehearsal2/`:
- `rehearsal_guarded/outputs/`: designs, carry records, participants, the inner loops'
  history, ledger, models, exports, candidates and token usage;
- `rehearsal_guarded/private/`: configuration, `prospective.json`, `recovery.json`;
- `ground_truth.json`;
- `logs/rsa_outer_rehearsal_47176690_0.out`.

Left out: trial-level CSVs (`private/existing_plain.csv` included), fit caches and agent
logs. Scanned for keys and Prolific ids; none.

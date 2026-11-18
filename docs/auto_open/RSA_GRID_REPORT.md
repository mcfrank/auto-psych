# Standard RSA and the campaigns' theories on every study: the Sherlock run (2026-11-18)

Written by Claude Opus 5.5 from the Sherlock run of `RSA_GRID.md` (steps 1-5); not reviewed by a person.
**Exploratory**: the closed campaign's pre-registered data are spent for confirmatory claims
(`docs/auto_rsa/CONSOLIDATION.md` §5), and most fits carry a convergence flag (§4).

**Bottom line.**
- **The best standard RSA is the simplest one with a salience prior:** depth 1, the top listener's prior favouring a
  singleton, Boolean meanings (2 parameters). It scores +2,318 nats over plain RSA, summed over 24 studies.
- **The grid's axes:**
  - Singleton salience carries almost all of the gain.
  - Feature-count salience adds about +420.
  - Recursion beyond depth 1 adds nothing.
  - Graded meanings *hurt* a depth-1 listener (−103, 0 of 5 pairs).
  - Whether the prior is shared across levels makes no difference.
- **The campaigns' best theories beat the best grid theory by about 1,500 nats.** These are the closed consolidation
  superset, closed chain 1's second incumbent and the fixed arm's second incumbent. Their gain is on pragmods, the
  closed experiments, stage 1 and the pilot. On Sikos 2021 and Mayn & Demberg 2026 the grid matches them.
- **Convergence:** 24 of the 36 grid fits and 11 of the 27 scored seed and discovered fits are flagged as not
  converged. A diagnostic traced the flags it examined to one nuisance link in `pilot1_exp2`. Leaving that study out
  changes no ranking that matters.
- **Gaps:** four closed incumbents timed out at 48 hours and are not scored. Closed chain 2's three theories lose
  about 14,000 nats on open pilot 1's bets, and nowhere else.

## 1. The run

- **Code** `auto-psych-open`:
  - fits at 5066bcde (grid) and d11953f4 (seed and discovered theories);
  - report and page at cd50702f. That version has the 36-theory grid (the cost axis was removed, PI 2026-11-17), the
    pragmods split under Sherlock's source name, and the page's disclosure line.
  - The 36 grid files were regenerated from cd50702f and checked byte-identical to the fitted ones. The 30 fits of
    the cost theories are left out.
- **Settings:** grouped 5-fold CV within every block (whole units per fold), 4 chains × 1,000 + 1,000, dense mass,
  target accept 0.9, seed 0.
- **Slurm:**

  | Theories | Array | CPUs, memory | Partition |
  |---|---|---|---|
  | Grid | 66 tasks (36 kept) | 2 CPUs, 15.6 GB each | `mcfrank`, 4-day limit |
  | Seeds and discovered theories | 31 tasks | 2 or 4 CPUs (the 11 closed theories 4) | `normal`, 48-hour limit |

  Per-trial `.npz` scores stay on Sherlock (`$SCRATCH/auto-psych/refits_1115/grid_fits/`).
- **Committed** in `data/rsa_open/grid/all_studies/`:
  - `grid_report.{md,json}`;
  - `grid_viz.html` (3.4 MB, 1,407 displays);
  - one `<theory>.json` per scored fit (converged flag, seconds).

### The corpus against `DATA_INVENTORY.md`

| block | trials | units | studies | inventory |
|---|---|---|---|---|
| E (literature) | 50,587 | 422 | pragmods 6,703; Sikos 2021 4,482; Mayn & Demberg 2026 16,764, 2023 15,048, 2022 7,590 | 50,587 ✓ |
| closed | 35,971 | 780 | nine chain experiments (2,145-2,509 each) and the final (14,404) | 35,971; 468 + 312 displays ✓ |
| pilot1_exp1 (bets) | 2,952 | 24 | | 2,952; 24 ✓ |
| pilot1_exp2 (choice + probed rating) | 4,530 | 30 | | 4,530; 30 ✓ |
| s1_open_exp1 / exp2 / exp3 | 4,750 / 3,925 / 6,840 | 50 / 50 / 60 | | ✓ |
| s1_fixed (one block) | 13,598 | 156 | exp1 4,602; exp2 4,394; exp3 4,602 | ✓ |
| **total** | **123,153** | **1,572** | 23 studies in the corpus, 24 report columns (pragmods split) | 123,153 ✓ |

The report splits pragmods into its plain displays (`pragmods_trials`, 5,390 trials) and the four manipulated-prior
experiments (`pragmods_prior`, 1,313), which the closed campaign's makers never saw.

## 2. The table

The out-of-fold elpd of each theory over plain RSA (`grid_d1_nosal_bool_nocost`), in nats:
- **pooled**: the sum over all 24 study columns.
- **pooled SE**: √(Σ per-study clustered SE²), which treats studies as independent. That understates it a little,
  since stage 1's chains share people.
- **without `pilot1_exp2`**: the same pooled score minus that study.
- **h**: fit hours.

Per-study cells are in `grid_report.md` and on the page.

| theory | pooled (SE) | without `pilot1_exp2` | converged | h |
|---|---|---|---|---|
| `cons_d2_dup_rich_…` (closed consolidation superset) | +3,824 (233) | +3,648 | yes | 15.3 |
| `closed_c1e2_contrastive_extremity_noisy_speaker` | +3,799 (219) | +3,596 | yes | 6.3 |
| `fixed2_contrastive_anchor_decision_l2` | +3,792 (255) | +3,597 | no | 2.8 |
| `cons_dup_sur_red_asp_int_comp_bind` (closed consolidation, lean) | +3,645 (220) | +3,460 | no | 33.3 |
| `fixed1_prototype_tier_confusion_l2` | +3,397 (237) | +3,188 | yes | 2.1 |
| `closed_c1e1_softmax_belief_listener` | +3,288 (214) | +3,103 | yes | 9.2 |
| `fixed3_sole_exclusion_shadowing_heuristic` | +3,224 (443) | +3,098 | yes | 1.9 |
| `pilot1_e2_attentional_buffer_anchor_sampling` | +3,053 (296) | +2,742 | yes | 2.9 |
| `rsa_l2_singleton_feat_color_valence_l0` (seed) | +2,975 (233) | +2,791 | yes | 2.8 |
| … two more `rsa_l2_singleton_*` seeds, `softmax_surprisal_isolated_costly_l4`, `pilot1_e1_…` | +2,381 to +2,909 | | | |
| **`grid_d1_lstsingleton_bool_nocost` (best grid)** | **+2,318 (190)** | **+2,233** | yes | 1.8 |
| `grid_d1_shrsingleton_bool_nocost` | +2,246 (191) | +2,165 | no | 3.8 |
| `grid_d2_lstsingleton_bool_nocost` | +2,138 (197) | +2,050 | no | 3.5 |
| … the other depth-1 to 3 singleton grid theories | +1,501 to +2,099 | | | |
| `grid_d0_lstsingleton_graded_nocost` (literal listener, singleton prior, graded) | +1,887 (184) | +1,798 | no | 1.7 |
| `open2_…`, `open3_…`, `competitor_confusion_speaker` | +1,752 to +1,787 | | no | |
| `open1_extension_noisy_graded_semantics_l2` | +664 (345) | +507 | yes | 4.1 |
| feature-count grid theories, depths 0-3 (12) | +363 to +419 | | | |
| plain RSA and the other no-salience theories, depths 1-3 | 0 to −30 | | | |
| `grid_d0_nosal_bool_nocost` (literal listener) | −4,727 (333) | −4,407 | yes | 1.8 |
| `closed_c2e1`, `c2e2`, `c2e3` (closed chain 2) | −10,335 to −10,962 (≈2,150) | | | |
| **not scored, timed out at 48 h:** `closed_c0e1`, `c0e2`, `c0e3`, `c1e3` | | | | 48 |

**Leaving `pilot1_exp2` out** (§4) lowers the leading theories' scores by 80-310 nats.
- It swaps `closed_c1e2` and `fixed2`, which are 1 nat apart either way.
- It also swaps the seed `rsa_l2_singleton_feat_color_valence_l0` and `pilot1_e2`.
- The best three grid theories keep their order.

**Closed chain 2's three theories** lose 13,435-14,314 nats on `pilot1_exp1`, open pilot 1's bet-allocation
experiment. They lose nothing comparable anywhere else. Without that one study they would score +3,100 to +3,350,
among the top eight.
- The pattern suggests their endpoint mechanisms give some objects nearly zero probability, which a forced choice
  tolerates and a bet on that object does not.
- That is an inference; I have not checked their code.
- No other theory loses more than 1,000 nats on any one study.

## 3. What the grid's axes do

Each row compares pairs of grid points that differ only on the named axis. All 24 studies are summed per pair, and
the row gives the mean over pairs. "Wins" counts pairs where the second level is better. Depth-0 pairs are left out
except for depth itself.

Pairs are **not** restricted to converged fits: two thirds of the grid is flagged, so that would leave one pair or none
on most rows. The few converged pairs are given beside each row. Leaving `pilot1_exp2` out changes the sign of no
contrast.

| contrast | pairs | mean | wins | converged pairs |
|---|---|---|---|---|
| depth 1 − 0 | 6 | +2,229 | 4/6 | 3: +3,216 |
| depth 2 − 1 | 10 | −59 | 3/10 | 1: +3 |
| depth 3 − 2 | 10 | −84 | 5/10 | 2: +1 |
| singleton salience − none, top listener | 6 | +2,139 | 6/6 | 1: +2,318 |
| singleton salience − none, shared | 6 | +1,940 | 6/6 | 0 |
| feature-count salience − none, top listener / shared | 6 / 6 | +417 / +422 | 6/6, 6/6 | 0 |
| singleton − feature count (same placement) | 12 | +1,620 | 12/12 | 0 |
| shared − top-listener salience (same kind) | 12 | −97 | 5/12 | 1: +0 |
| graded − Boolean, depth 1 | 5 | −103 | 0/5 | 0 |
| graded − Boolean, depth 2 / 3 | 5 / 5 | −30 / −29 | 3/5, 2/5 | 1 / 1: +1 / −0 |
| (graded − Boolean, depth 0: the literal listener) | 3 | +4,237 | 3/3 | 0 |

How this compares with the literature-only run (`RSA_GRID.md`, three committed sources):

1. **Singleton salience dominates even more.** It is worth about 2,000 nats here against about 400 there. It helps on
   nearly every study, not only Sikos 2021.
2. **Feature-count salience now helps:** +420, in 12 of 12 pairs, where on the literature it lost in every pair.
   The closed campaign's and stage 1's displays carry that signal.
3. **Depth beyond 1 adds nothing.**
   - The negative means for 2 − 1 and 3 − 2 come from pairs with a flagged fit.
   - The converged pairs are level.
   - The two depth-3 theories with a shared singleton prior score 400-550 nats below their depth-2 neighbours. That is
     the same fitting failure the literature run found: a prior repeated through four levels makes the salience
     weight and alpha hard to separate.
4. **Graded meanings no longer help a depth-1 listener.**
   - They lose in 5 of 5 pairs, against +41 in 5 of 5 on the literature.
   - At depth 0 they are still a large repair, as they were on the literature.
   - On this corpus the open arm's recurring discovery is not a free gain for standard RSA.
5. **Whether salience is common knowledge** makes no difference in the one converged pair.

## 4. Convergence, and what the flags mean

**Not converged:** a fold fit above 0.1% divergent transitions, R-hat 1.05 or below 100 bulk ESS.
- **Grid, 24 of 36:**
  - 3 of the 6 at depth 0;
  - 7 of the 10 at each of depths 1, 2 and 3;
  - among them 5 of the 6 no-salience theories at depths 1-3, all but plain RSA.
- **Seed and discovered, 11 of 27:** `rsa_l1_shared_prior`, `isolated_graded_costly_l3`,
  `rsa_l2_singleton_feat_color_familiar_l0`, `competitor_confusion_speaker`, `softmax_surprisal_isolated_costly_l4`,
  `closed_c2e1`, `closed_c2e2`, `cons_dup_sur_red_asp_int_comp_bind`, `open2_…`, `open3_…` and `fixed2_…`.

**The diagnostic** re-fitted three flagged theories (`grid_d0_nosal_graded_nocost`, a depth-1 cost grid theory and
`competitor_confusion_speaker`) with the same data, folds, settings and seed, recording every fold.
- Every failing fold failed the same way, in 3, 2 and 2 of 5 folds: one of the four chains stuck on the rating link of
  `pilot1_exp2`'s second response. Its centre, spread and lapse had R-hat 1.53 and bulk ESS 7, identical in the
  three theories.
- There were no divergences.
- The theories' own parameters and every other study's link converged in every fold (R-hat at most 1.004, bulk ESS at
  least 3,600).
- That link's parameters belong to `pilot1_exp2` alone, so a stuck chain there should bias only that study's scores.
- The tables give every score with and without it, and the conclusions above do not depend on it.

**What is not established:**
- **Other causes in the 33 flagged fits not re-fitted.** The diagnostic covered three, two of them in this set; the
  other 33 may also have other problems. The depth-3 shared-singleton theories are the clearest sign of a genuine failure (§3, point 3).
- **Unfittable theories.** A theory that cannot be fitted in 48 hours on 4 CPUs is a cost in its own right, whatever
  the cause.
- **A fix.** Running that link with a different parameterisation, or more warmup, would settle it but needs re-fits
  (about a day). It has not been done.

## 5. Each study's best theory

| study | best theory | over plain RSA | best grid theory | over plain RSA |
|---|---|---|---|---|
| Mayn & Demberg 2022 | `fixed3` | +226 | d2, top-listener singleton, Boolean | +69 |
| Mayn & Demberg 2023 | `open1` | +88 | d2, top-listener singleton, graded | +22 |
| Mayn & Demberg 2026 | `closed_c1e2` | +105 | d2, top-listener singleton, Boolean | +86 |
| pragmods, plain | `fixed3` | +176 | d3, shared singleton, Boolean | +124 |
| pragmods, prior manipulated | `pilot1_e2` | +159 | d3, shared singleton, graded | +14 |
| Sikos 2021 | `fixed2` | +319 | d2, top-listener singleton, Boolean | +316 |
| closed chain experiments (9) | `fixed3` on 8, `fixed1` on 1 | +111 to +182 | singleton theories, depths 0-2 | +51 to +88 |
| closed final | `fixed3` | +847 | d1, top-listener singleton, Boolean | +507 |
| pilot 1 exp 1 (bets) | `pilot1_e1` | +257 | d3, shared singleton, Boolean | +116 |
| pilot 1 exp 2 | `pilot1_e2` | +311 | d2, shared singleton, graded | +91 |
| stage 1 open exp 1 (choice and rating) | `open1` | +243 | d0, top-listener singleton, graded | +39 |
| stage 1 open exp 2 (two ratings) | a grid theory (d2, top-listener singleton, graded) | +57 | the same | +57 |
| stage 1 open exp 3 (production) | `open3` | +426 | **d0, no salience: the literal listener** | +347 |
| stage 1 fixed exp 1-3 | `fixed3` | +364 to +482 | singleton theories, depths 0-3 | +208 to +324 |

What stands out:
- **The fixed arm's final (`fixed3`) is the best theory on 14 of the 24 study columns.** Three are its own chain's
  experiments. The other 11 are data its makers never saw: eight closed chain experiments, the closed final, Mayn &
  Demberg 2022 and plain pragmods. That is the same theory
  whose cross-refit fit was flagged, and whose re-fit converged (`CROSS_REFIT_REPORT.md`). It converged here.
- **On production, a literal listener's speaker beats plain RSA by 347 nats.** Only the open arm's final does better.
- **On the dual-rating experiment, no campaign theory beats the grid.**

## 6. Time

Fit seconds from each theory's json; the four timeouts from `sacct`.

| set | n | min | median | max (h) |
|---|---|---|---|---|
| grid, depth 0 | 6 | 1.3 | 1.9 | 2.1 |
| grid, depth 1 | 10 | 1.5 | 3.0 | 14.4 |
| grid, depth 2 | 10 | 1.8 | 2.7 | 15.5 |
| grid, depth 3 | 10 | 1.9 | 3.3 | 31.9 |
| seed and discovered, scored | 27 | 1.3 | 4.0 | 33.3 |
| seed and discovered, timed out | 4 | 48 | | |

- No converged fit took longer than 15.3 hours, and every fit that ran past 15.5 hours was flagged.
- Long fits go with the stuck-chain pattern. In the cross refit's diagnostic, one fold's warmup took 15 times the
  usual leapfrog steps (`CROSS_REFIT_REPORT.md`).
- The closed theories were slowest. That was expected (chain 0's final has 17 parameters), but the four timeouts are
  chain 0's three incumbents and chain 1's third.

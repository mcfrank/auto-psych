# Experiment 2 Design Notes (chain_0, cycle_1)

## 1. Context & State of the Theories after Experiment 1

### Experiment 1 Summary (`graded_belief_allocation`)
- **Task**: 10-point belief allocation (`bet` / `allocation`) across 3-4 objects and 2-3 features (`sizes: [[3,2], [3,3], [4,3]]`, `prior_queries: true`, `synonyms: true`).
- **Held-out LPD fits (`1_fits.md`)**:
  1. `bounded_depth_listener`: **best**
  2. `rsa_l1_singleton_feat_color_valence_l0`: **5.1 ± 17.2 nats** behind best (statistically indistinguishable)
  3. `bounded_depth_feature_parsimony`: **12.3 ± 9.8 nats** behind best
  4. `rsa_l2_singleton_feat_color_valence_l0`: **42.3 ± 37.5 nats** behind best

## 2. Candidate Exploration & Quantitative Comparisons

We systematically compared candidate paradigms across response types (`production`, `rating`, `slider`, `choice`) and scene spaces (`sizes`) using `validate`, `eig --quick`, `eig`, and `link-check`:

1. **`cand_production.json` (`production` link, 2-4 objects, 2-4 features)**:
   - `eig --quick`: 3.525 of 4.170 bits (`0.02644 bits/$`, meets floor `0.01271`), estimated 2.6 min vs. declared 3.0 min.
   - `link-check` (on 3-feature version): `converged=True, recovered=True`, 1 miss on `w_features` because `production` has `prior_queries: false`.
2. **`cand_choice.json` (`choice` link, 2-5 objects, 2-4 features, excluding `[2, 4]`)**:
   - Note: `[2, 4]` has no non-synonym valid games since $2^2 - 1 = 3 < 4$, so we excluded `[2, 4]` from all 4-feature scene spaces.
   - `eig --quick`: 3.623 of 4.170 bits (`0.02717 bits/$`, meets floor `0.01271`), estimated 2.9 min vs. declared 3.0 min.
   - `link-check`: `converged=True, recovered=True`, **0 misses across all 8 parameters**.
3. **`cand_slider.json` (`ordered_beta` link, 2-5 objects, 2-3 features)**:
   - `eig --quick`: 3.912 of 4.170 bits (`0.02934 bits/$`), estimated 3.3 min vs. declared 3.0 min.
4. **`probed_referent_likelihood_rating.json` (`rating` link, 7-point scale, `sizes: [[2, 2], [2, 3], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [5, 2]]`, `prior_queries: true`, `synonyms: true`)**:
   - **Key finding on `link-check`**: When 4-feature scenes (`[3, 4], [4, 4]`) were omitted and `[5, 3]` dominated the random pool (`cand_rating.json` v1), `alpha` and `temperature` exhibited a slight collinearity ridge. Including 4-feature scenes (`[3, 4], [4, 4]`) and replacing `[5, 3]` with `[5, 2]` completely broke the `alpha`–`temperature` collinearity because 4-feature scenes create utterances with extensions $|u| \in \{1, 2, 3, 4\}$ whose nonlinear sum in $S_1$'s denominator cannot be mimicked by `temperature`.
   - **Validation (`sha256: d59103b863d7a99f870ea08ec74f2d049a349773a9962c53f0f74e54dfe38ded`)**:
     - Valid: 1,649 scenes (5,275 trial units), `rating` via `rating`, `$133.33` (`n_participants=200, trials_per_participant=20, distinct_trials=50, catch_trials=2, minutes=3.0, pay_usd=0.50`).
     - Time estimate: `3.1 min` against declared `3.0 min` (within 3.3%, well inside the 30% pilot tolerance).
   - **Full `link-check` (`work/reports/link-check_d59103b863d7.json`)**:
     - `converged = True, recovered = True`, **0 misses across all 10 parameters** (`alpha: 0, w_singleton: 0, w_distinct: 0, w_features: 0, w_valence: 0, w_color: 0, center: 0, spread: 0, lapse: 0, temperature: 0`).
   - **Full `eig` (`work/reports/eig_d59103b863d7.json`)**:
     - `joint_eig_bits`: **3.913 of 4.170 bits** (`p_correct = 0.831` overall across all 18 models, and `p_correct = 1.000` with `mean_posterior_on_truth = 1.0000` on `bounded_depth_listener`, `rsa_l1_singleton_feat_color_valence_l0`, and `bounded_depth_feature_parsimony`).
     - `bits_per_dollar`: **0.02934 bits/$** against floor **0.01287 bits/$** (**228% of the floor**; reference paradigm is `0.02574 bits/$`).

## 3. SVG Materials Verification
- Domain 1 (`snowmen`): complete 3-snowball snowman base with eyes, carrot nose, smile, and buttons; 4 spatially non-overlapping features:
  - `hat` (top of head, `x=34..66, y=4..22.5`)
  - `scarf` (neck, `x=36..63, y=38..60`)
  - `broom` (left side, `x=6..37, y=16..90`)
  - `balloon` (right side, held by right arm with string and purple oval balloon + gloss highlight, `x=63.5..94, y=2.5..50`)
- Domain 2 (`houses`): complete house base with walls, pitched roof, door with knob, and two paned windows; 4 spatially non-overlapping features:
  - `chimney` (right roof slope, red brick chimney + smoke puffs, `x=60..77, y=4..38`)
  - `flag` (left yard, pole + blue triangle flag, `x=7..23, y=26..88`)
  - `tree` (right yard, green pine tree, `x=77..99, y=44..88`)
  - `sun` (top-left sky above roof, yellow sun with 8 rays, `x=24..44, y=2..22`)

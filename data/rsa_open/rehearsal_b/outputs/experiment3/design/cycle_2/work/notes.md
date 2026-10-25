# Experiment 3 Design Notes (`chain_0`, `cycle_2`)

## 1. Diagnosis of Previous Cycle (`cycle_1`) Failure
- In `cycle_1`, the designer attempted to submit `probed_referent_probability_slider.json` (`slider` / `ordered_beta` link).
- While `validate` and `eig` succeeded (taking ~27 minutes total), `link-check` on `ordered_beta` (which fits 5 link parameters `cut_low`, `cut_gap`, `phi`, `lapse`, `temperature` + 6 theory parameters = 11 parameters with discrete Beta softmax likelihoods across 9 shape groups) hung in NUTS for >93 minutes until the 120-minute session wall-clock limit terminated the cycle.
- Lesson: Avoid `ordered_beta` (`slider`). Use fast, clean links (`choice` has 2 link parameters `lapse` and `temperature`, and `link-check` completes in ~9 minutes with 0 misses).

## 2. Candidate Exploration in `cycle_2`
We compared candidate paradigms across response types and scene spaces using `validate` and `eig --quick`:
1. **`explore_choice.json` (`choice` link, 2–5 objects, 2–4 features, $N=200 \times 22 = 4{,}400$ trials, $D=66$)**:
   - `eig --quick`: `3.461` of `4.087` bits (`p_correct = 0.74`), `0.02595 bits/$` (floor `0.01188`), estimated time `2.8 min` vs declared `3.0 min`.
2. **`explore_production.json` (`production` link, 2–4 objects, 2–4 features, $N=200 \times 22 = 4{,}400$ trials, $D=66$)**:
   - `eig --quick`: `3.335` of `4.087` bits (`p_correct = 0.77`), `0.02502 bits/$` (floor `0.01188`), estimated time `2.8 min` vs declared `3.0 min`.
   - However, because none of the 5 live theories define a custom `speaker()`, derived speaker distributions $P_S(u \mid r) \propto P_L(r \mid u)$ cancel out target-object priors and feature-parsimony penalties in the numerator, yielding low identification on `bounded_depth_listener` (`p_correct = 0.50`) and `bounded_depth_feature_parsimony` (`p_correct = 0.42`).
3. **`explore_bet_valid.json` (`allocation` link, 2–5 objects, 2–4 features, $N=150 \times 16 = 2{,}400$ trials, $D=48$, `minutes=3.8, pay_usd=0.66`)**:
   - `eig --quick`: `3.632` of `4.087` bits (`p_correct = 0.70`), `0.02751 bits/$`. Moreover, Experiment 1 in `chain_0` already tested 10-point `bet` / `allocation`.
4. **`multi_competitor_referent_choice.json` (`choice` link, 2–6 objects, 2–4 features: `[[2, 2], [2, 3], [3, 2], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [5, 2], [6, 2]]`, `prior_queries: true`, `synonyms: true`, $N=200 \times 28 = 5{,}600$ trials, $D=56$, `minutes=3.0, pay_usd=0.50`)**:
   - **SHA256**: `3982e7ef618cf8d08de91a38b20573ad9b7aab5869dc79b96920cd4b9b1a140a`
   - **Validate**: Valid, 1,772 scenes (1,772 trial units), `$133.33`, estimated `3.3–3.4 min` against declared `3.0 min` (within 30% tolerance).
   - **Full `link-check` (`work/reports/link-check_3982e7ef618c.json`)**:
     - `converged = True, recovered = True`
     - `misses = {'alpha': 0, 'w_singleton': 0, 'w_distinct': 0, 'w_features': 0, 'w_valence': 0, 'w_color': 0, 'lapse': 0, 'temperature': 0}` (0 misses across all 8 parameters).
   - **Full `eig` (`work/reports/eig_3982e7ef618c.json`)**:
     - `joint_eig_bits`: **3.491 of 4.087 bits** (`p_correct = 0.792`, vs `3.263` bits and `p_correct = 0.714` for the reference paradigm).
     - `bits_per_dollar`: **0.02618 bits/$** against floor **0.01224 bits/$** (**214% of the floor**).
     - Per-theory accuracy on the 5 live theories:
       - `rsa_l1_singleton_feat_color_valence_l0`: `p_correct = 0.966` (`mean_posterior = 0.930`)
       - `feature_uncertainty_listener_2`: `p_correct = 0.938` (`mean_posterior = 0.921`)
       - `bounded_depth_listener`: `p_correct = 0.767` (`mean_posterior = 0.635`)
       - `bounded_depth_feature_parsimony`: `p_correct = 0.700` (`mean_posterior = 0.674`)
       - `rsa_l2_singleton_feat_color_valence_l0`: `p_correct = 0.625` (`mean_posterior = 0.419`)

## 3. Materials & Environment Notes
- Materials use the pilot-verified 4-feature `snowmen` (`hat`, `scarf`, `broom`, `balloon`) and `houses` (`chimney`, `flag`, `tree`, `sun`) SVG domains (which achieved 100% legibility, 16/16 items, 3/3 checkers in Experiment 2's pilot).
- `work/sitecustomize.py` provides `src.rsa.experiment.design._catch_spec` when running CLI tools inside the scrubbed agent tree with `PYTHONPATH=/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment3/design/cycle_2/work:.`.

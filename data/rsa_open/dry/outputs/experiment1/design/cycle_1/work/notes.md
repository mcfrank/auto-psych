# Designer Notes — Chain 0, Experiment 1 (Cycle 1)

## 1. Environment & Tooling Note
- In the scrubbed agent sandbox (`/scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/repo`), `src/rsa/experiment` was scrubbed out, while `src/rsa_open/screens.py` imports `_catch_spec` from `src.rsa.experiment.design` during `time_report`.
- `/tmp/sitecustomize.py` provides `src.rsa.experiment.design._catch_spec` when running `PYTHONPATH=/tmp /scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/venv/bin/python -m src.rsa_open.tools ...`. (The parent harness process runs in the full repository where `src.rsa.experiment.design` is present.)

## 2. Theory & Posterior Analysis
Posterior means (+/- SD) from `context/draws/*.npz`:
- **Live theories (6):**
  1. `competitor_confusion_speaker`: `alpha = 0.644 +/- 0.111`, `w_confusion = 1.379 +/- 0.144`
  2. `isolated_graded_costly_l3`: `alpha = 0.921 +/- 0.157`, `w_distinct = 0.330 +/- 0.053`, `gamma = 0.051 +/- 0.692`, `cost_weight = -0.511 +/- 0.599`
  3. `oddity_heuristic_listener`: `beta = 0.240 +/- 0.106`
  4. `rsa_l2_singleton_feat_color_valence_l0`: `alpha = 1.698 +/- 0.087`, `w_singleton = 0.456 +/- 0.078`, `w_distinct = 0.361 +/- 0.032`, `w_features = 0.223 +/- 0.040`, `w_valence = 0.956 +/- 0.080`, `w_color = 0.561 +/- 0.129`
  5. `rsa_l2`: `alpha = 0.776 +/- 0.038`
  6. `rsa_l1`: `alpha = 1.322 +/- 0.084`
- **Reference/bar theories in context (9):**
  - `literal_listener`, `rsa_l1_salience`, `rsa_l1_shared_prior`, `familiar_isolated_graded_costly_l2`, `rsa_l2_singleton_distinct_feat_color_l0`, `evaluative_prominence_l2`, `lexical_preemption_heuristic`, `rsa_l2_singleton_feat_color_familiar_l0`, `softmax_surprisal_isolated_costly_l4`.

## 3. Paradigm Selection & Evaluation
- **Final spec**: `work/paradigm_bet_allocation.json` (`sha256`: `5d717313007e3d36c2017278c9347bb78e3461a5b214957a29098928bc977542`)
- **Response type & link**: `bet` (`allocation` link, `total_points = 10`).
- **Scene space**: `[[2,2],[2,3],[2,4],[3,2],[3,3],[3,4],[4,2],[4,3],[4,4]]`, `prior_queries = true`, `synonyms = true` (1,597 scenes).
- **Design**: `n_participants = 100`, `trials_per_participant = 18`, `distinct_trials = 45` (40 responses/unit, 1,800 total test responses), `catch_trials = 2`, `minutes = 5.1`, `pay_usd = 0.85` (`cost_usd = $113.33`).
- **Full EIG (`reports/eig_5d717313007e.json`)**:
  - `joint_eig_bits = 3.551` of `3.907` bits (`p_correct = 0.711`), exceeding the closed campaign reference (`3.261` bits, `p_correct = 0.691`).
  - `bits_per_dollar = 0.03133` bits/$ vs. `floor_bits_per_dollar = 0.01223` bits/$ (**256% of floor**, `meets_floor = true`).
  - `estimated_minutes = 5.107` min vs. `declared_minutes = 5.1` min (`within_tolerance = true`, 0.14% difference).
- **Link recovery (`reports/link-check_5d717313007e.json`)**:
  - Simulated from `rsa_l2_singleton_feat_color_familiar_l0` (1,800 responses, 3 replicates): `converged = true`, `recovered = true`, `misses = 0` across all 9 parameters (`alpha`, `w_singleton`, `w_distinct`, `w_features`, `w_color`, `w_familiar`, `kappa`, `lapse`, `temperature`).
- **Materials**:
  - 3 SVG domains (`houses`, `robots`, `snowmen`), 4 spatially non-overlapping features each, verified via `resvg_py` rendering.

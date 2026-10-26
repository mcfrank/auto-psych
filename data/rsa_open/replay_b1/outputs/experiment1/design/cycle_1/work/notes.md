# Designer Session Notes (`chain_0`, Experiment 1, Cycle 1)

## Key Findings from Exploration
1. **Posterior Draws & Theoretical Landscape**:
   - Live theories: `competitor_confusion_speaker`, `isolated_graded_costly_l3`, `oddity_heuristic_listener`, `rsa_l2_singleton_feat_color_valence_l0`, `rsa_l2`, `rsa_l1`.
   - Reference paradigm EIG (cached in `context/reference_e75ce4fb11bc_64_1000_0_lazy16.json`): `3.21085` bits out of `3.90689` bits (`0.02408` bits/$, floor `0.01204` bits/$).
   - Note that three theories in the 15-theory context pool (`rsa_l2_singleton_distinct_feat_color_l0`, `rsa_l2_singleton_feat_color_familiar_l0`, `rsa_l2_singleton_feat_color_valence_l0`) differ only on `familiarization` and `valence` covariates, which are 0 in the open `context_pool`, capping maximum theoretical EIG across all 15 models at ~3.59 bits.

2. **Submitted Paradigm (`work/graded_belief_allocation.json`)**:
   - SHA256: `c1abdfcaee35f28cb48de3c6aea80292dd1ae47401f0368bf27c2e2456f5e239`
   - Response type: `bet` via `allocation` link (`total_points: 10`).
   - Scene space: `sizes: [[3, 2], [3, 3], [4, 3]]`, `prior_queries: true`, `synonyms: true` (330 scenes, 3 shape groups for fast JAX/memo compilation and NUTS fitting).
   - Design: `n_participants: 120`, `trials_per_participant: 15`, `distinct_trials: 45` (40 responses/unit, 1800 total test responses), `catch_trials: 2`, `minutes: 3.9`, `pay_usd: 0.65` ($104.00 total cost).
   - Full `eig` report (`work/reports/eig_c1abdfcaee35.json`):
     - `joint_eig_bits`: **3.477** of 3.907 bits (`p_correct`: **0.77**)
     - `bits_per_dollar`: **0.03344** bits/$ (floor `0.01204`: **MEETS**, 2.78x the floor)
     - `time`: estimated **4.1 min** vs declared **3.9 min** (within 5%, well inside the 30% tolerance).
   - Full `link-check` report (`work/reports/link-check_c1abdfcaee35.json`):
     - `converged`: `True`
     - `recovered`: `True`
     - `misses`: `{'alpha': 0, 'w_singleton': 0, 'w_distinct': 0, 'w_features': 0, 'w_color': 0, 'w_familiar': 0, 'kappa': 0, 'lapse': 0, 'temperature': 0}` (0 misses across all 9 parameters!).
   - Materials: 2 SVG domains (`snowmen`: `hat`, `scarf`, `broom`; `houses`: `chimney`, `flag`, `tree`), visually verified via `resvg_py` in `work/preview_materials.png`.

3. **Environment Note**:
   - In the scrubbed agent tree, `src/rsa/experiment` was omitted, so `work/sitecustomize.py` provides `src.rsa.experiment.design._catch_spec` when running CLI tools with `PYTHONPATH=/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/design/cycle_1/work`.

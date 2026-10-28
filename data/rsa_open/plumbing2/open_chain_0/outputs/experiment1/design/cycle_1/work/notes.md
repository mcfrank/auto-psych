# Design Cycle 1 Notes — Chain 0, Experiment 1

## Exploration & Tool Results
1. **Theory Analysis (`context/theories/*.py`)**:
   - All 15 theories (6 live + 9 bar) define `listener(params, ctx)` over `ctx.lex`, `ctx.utterance`, `ctx.is_prior`, `ctx.feature_count`, `ctx.is_sink`, `ctx.grayscale`, `ctx.familiarization`, `ctx.valence`.
   - Live theories:
     - `competitor_confusion_speaker`: Depth-1 listener inverting a speaker with confusion penalty `-w_confusion * (real_lex @ exp(-dist))` and prior `softmax(-w_confusion * sum(exp(-dist)))`.
     - `isolated_graded_costly_l3`: Depth-3 listener (`L3`) with minimum Hamming distance prior (`object_isolation`), graded semantic truth (`exp(-gamma * (fc - 1))`), and utterance extension cost (`cost_weight * ext`).
     - `oddity_heuristic_listener`: Non-recursive heuristic weighting literal matches by `exp(beta * is_singleton)`. Collapses to uniform literal choice whenever all objects are distinct (`is_singleton = 1` for all objects).
     - `rsa_l2_singleton_feat_color_valence_l0`: Depth-2 listener (`L2`) with literal `L0` and common-knowledge perceptual prior (`w_singleton * is_singleton + w_distinct * sum_hamming_dist + (w_features + valence * w_valence) * feature_count + w_color * (1 - grayscale)`).
     - `rsa_l2`: Depth-2 RSA with uniform prior.
     - `rsa_l1`: Depth-1 RSA with uniform prior.

2. **Candidate Paradigm Comparison (`eig --quick` & `link-check`)**:
   - `exp_choice_baseline.json` (`choice` via `choice`, `[[3,3],[4,3],[4,4]]`, `prior_queries=True`, $120):
     - `eig --quick`: 3.282 / 3.907 bits (`p_correct = 0.52`), 0.02735 bits/$ (floor 0.00650).
   - `exp_production.json` (`production` via `production`, `[[3,3],[4,3],[4,4]]`, `synonyms=True`, $120):
     - `eig --quick`: 3.185 / 3.907 bits (`p_correct = 0.53`), 0.02654 bits/$.
   - `exp_choice_second_rating.json` (`choice` + `second_response: rating`, $140):
     - `eig --quick`: 3.683 / 3.907 bits (`p_correct = 0.60`), 0.02631 bits/$.
   - `exp_bet.json` / `graded_belief_allocation.json` (`bet` via `allocation`, 10 points, `[[3,3],[4,3],[4,4]]`, `prior_queries=True`, `n_participants=150`, `trials_per_participant=24`, `distinct_trials=48`, 6.0 min, $200):
     - `eig --quick`: **3.742 / 3.907 bits** (`p_correct = 0.595`), 0.01871 bits/$ (nearly 3x floor of 0.00650).
     - `link-check`: **PASSED** (`converged=True, recovered=True`, `misses={'alpha': 0, 'w_singleton': 0, 'w_distinct': 0, 'w_features': 0, 'w_color': 0, 'w_familiar': 0, 'kappa': 1, 'lapse': 0, 'temperature': 0}`).
   - `exp_bet_wide.json` (adding 5-object scenes) and `exp_bet_synonyms.json` (adding `synonyms=True`):
     - `eig --quick`: 3.726 bits and 3.734 bits, slower per trial (7.4-8.1 min). Thus `[[3,3],[4,3],[4,4]]` with `prior_queries=True, synonyms=False` is both faster/cleaner and achieves the highest EIG (3.742 bits).

3. **Selected & Submitted Paradigm (`graded_belief_allocation.json`)**:
   - SHA256: `b0b54591df50` (validated at 6.0 min estimated vs 6.0 min declared).
   - Full `eig` (`reports/eig_b0b54591df50.json`): **3.737 of 3.907 bits** (`p_correct = 0.74`), `$200.00`, **0.01869 bits/$** (floor `0.00643`: MEETS, 2.91x floor; exceeds reference forced-choice EIG of 3.427 bits and 0.01285 bits/$).
   - Full `link-check` (`reports/link-check_b0b54591df50.json`): **PASSED** (`converged=True, recovered=True`, `misses={'alpha': 0, 'w_singleton': 0, 'w_distinct': 0, 'w_features': 0, 'w_color': 0, 'w_familiar': 0, 'kappa': 0, 'lapse': 0, 'temperature': 0}`).
   - Materials: SVG `robot` domain with complete, unmistakable metallic robot base (head with side ear-bolts, cyan LED eyes, segmented mouth grille, torso with corner rivets, arms with C-shaped pincer claws, legs and feet) and 4 spatially disjoint, high-contrast features: `hat` (red/gold top hat, y=3..19.5), `glasses` (blue eyeglasses over eyes, y=22..31), `scarf` (green neck scarf with right-side fringed tail, y=37..60), and `belt` (brown waist belt with gold buckle, y=63.5..74.5).
   - Submitted via `src.rsa_open.tools submit`: `submitted b0b54591df50`.

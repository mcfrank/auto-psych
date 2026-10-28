# Experiment 2 Design Cycle 1 Notes

## Posterior Draws Analysis (`context/draws/*.npz`)
- `rsa_l2_singleton_feat_color_valence_l0`: `alpha = 1.243±0.041`, `w_singleton = 1.008±0.031`, `w_distinct = 0.226±0.008`, `w_features = 0.010±0.008`, `w_valence = 1.048±0.087`, `w_color = 0.552±0.121`.
- `isolated_graded_costly_l3`: `alpha = 0.753±0.042`, `w_distinct = 0.527±0.012`, `gamma = -2.945±0.390`, `cost_weight = -0.571±0.116`.
- `competitor_confusion_speaker`: `alpha = 0.183±0.305`, `w_confusion = 0.786±0.061`.

## Key Theoretical Insight
In Experiment 1 (`cand_bet`, 2-4 objects, mean 3.44 objects/trial), `rsa_l2_singleton_feat_color_valence_l0` (`w_distinct = 0.226` on sum of Hamming distances) and `isolated_graded_costly_l3` (`w_distinct = 0.527` on min Hamming distance) were collinear because `0.527 / 0.226 = 2.33 ≈ N - 1`. Expanding the scene space to include 5- and 6-object scenes (`[5,3]`, `[6,3]`) breaks this collinearity completely and tests whether `isolated_graded_costly_l3`'s extreme negative `gamma = -2.945` generalizes beyond 10-point betting.

## Exploration & Tool Runs
1. `cand_test2_choice_rating.json` (`choice` + `second_response: rating`, sizes `[2,2]..[6,3]`):
   - `eig --quick`: `3.447 of 3.907 bits (p_correct 0.60), $266.40, 0.01294 bits/$; floor 0.00557: MEETS`.
2. `cand_test3_prod.json` (`production`, sizes `[2,2]..[6,3]`):
   - `eig --quick`: `3.089 of 3.907 bits (p_correct 0.57), $266.67, 0.01158 bits/$; floor 0.00557: MEETS`.
3. `cand_test1.json` (`choice`, sizes `[2,2]..[6,3]`):
   - `eig --quick`: `3.192 of 3.907 bits (p_correct 0.54), $266.67, 0.01197 bits/$; floor 0.00557: MEETS`.
4. `choice_probed_rating.json` (`choice` + `second_response: rating`, 3404 scenes / 11980 trial units):
   - Full `eig` passed: `3.304 of 3.907 bits (p_correct 0.68), $266.40, 0.01240 bits/$; floor 0.00567: MEETS` (report `eig_1dfa6e1c097c.json`).
   - However, `link-check` on 10 scene shapes + 6-parameter `pair` link exceeded 30 min timeout due to MCMC cost across 11,980 units / 10 shapes.
5. `multiscale_choice.json` (`choice`, 6 scene sizes `[[2,2],[3,2],[3,3],[4,3],[5,3],[6,3]]`, 2181 trial units, 2 link parameters):
   - Full `eig` passed in 5 minutes: `3.196 of 3.907 bits (p_correct 0.68), $266.67, 0.01199 bits/$; floor 0.00567: MEETS` (report `eig_db48c6a6d888.json`).
   - Both `choice_probed_rating.json` (`0.01240 bits/$`, `eig_1dfa6e1c097c.json`) and `multiscale_choice.json` (`0.01199 bits/$`, `eig_db48c6a6d888.json`) have passing full `eig` reports in `work/reports/` well above the floor (`0.00567 bits/$`) and matching time estimates (`3.6 min` and `3.0 min vs 2.7 min`).


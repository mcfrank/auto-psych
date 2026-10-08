This model refines fewest_features_listener by replacing raw feature counts with contextual feature rarity: listeners penalize candidate referents not for total visual complexity, but specifically for possessing extraneous features that are distinctive and rare in the display. When interpreting a referring expression, an extraneous unmentioned feature incurs a heavy penalty if it is uniquely distinguishing in the visual scene because a speaker could have used it as an unambiguous alternative, whereas extraneous features widely shared across competitors incur minimal penalty. This improves model fit because human referent choice is sensitive to the communicative contrast of unmentioned features rather than raw perceptual part counts.

Differences from fewest_features_listener:
- Refined model: fewest_features_listener
- Recursion depth: depth 0 (choice_probs calls L_heuristic, unchanged).
- Parameters added: none.
- Parameters removed: none.
- Contextual feature rarity: choice_probs computes a contextual rarity score for each candidate referent by weighting each present feature inversely by the number of objects sharing it in the visual scene (taking the contextual rarity component from feature_rarity_prior).
- Heuristic penalty in L_heuristic: candidate referents matching the uttered word are penalized in proportion to their contextual rarity score rather than raw feature count (vec(rarity_score, r) replaces vec(feature_count, r)).
- All other memo agents, distributions, and prior terms are unchanged.

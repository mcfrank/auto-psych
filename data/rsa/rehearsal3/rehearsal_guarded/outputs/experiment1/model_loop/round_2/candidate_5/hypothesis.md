Listeners interpret referring expressions by modeling a depth-2 speaker who balances communicative informativeness against competitor confusion, but who occasionally makes unintended speech production errors (a trembling hand). When a heard word is ambiguous and provides poor communicative utility for matching targets, listeners attribute the utterance to an accidental production slip, naturally explaining why listeners sometimes choose non-matching foil objects under ambiguous descriptions while remaining exceptionally accurate on unambiguous trials. Uninformative prior trials remain governed by visual singleton pop-out, distinctiveness, and feature complexity.

Model refined: `r1_c4`.
Differences from source:
- Recursion depth: Unchanged (depth 2, calling L2 in choice_probs).
- Parameters added: `tremble` (prior Beta(1.0, 19.0)) representing speaker speech production error rate.
- Parameters removed: None.
- Speaker policy: Extended at speaker tier S2 to incorporate trembling-hand speech production errors with rate `tremble` over real vocabulary words (`(1.0 - tremble) * s2 + tremble * tremble_dist`), which pragmatic listener L2 inverts.
- Referent prior: Unchanged (combining discrete singleton salience, continuous contextual distinctiveness, visual feature count modulated by framing valence, and color contrast).

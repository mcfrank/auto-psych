We refine rsa_l2 by incorporating visual singleton salience into the common-ground prior over referents. When interpreting ambiguous referring expressions or choosing without an informative word, listeners and speakers reasoning at depth 2 expect unique objects that lack identical visual duplicates in the display to be preferred over duplicated items. On uninformative trials, listeners spontaneously select the contextually unique singleton, defaulting to a uniform distribution when all items in the display are distinct.

Differences from rsa_l2:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_singleton (Normal(0.0, 1.0), weighting the visual singleton salience prior).
- Parameters removed: None.
- Other terms: In L1 and L2, the speaker's prior distribution over referents is changed from uniform (wpp=1) to the singleton salience prior (wpp=vec(prior, r)), where prior is softmax-weighted by w_singleton * is_singleton; on uninformative prior trials, the listener's choice is changed from uniform guessing to this singleton salience prior; uniform guessing is mixed in via the lapse parameter.

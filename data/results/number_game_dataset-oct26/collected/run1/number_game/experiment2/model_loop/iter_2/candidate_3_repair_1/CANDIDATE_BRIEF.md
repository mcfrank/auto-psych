# Candidate Brief

Refine the incumbent: `decade_band_digit_common_factor` — rank 0, the best model on this data, ELPD-LOO -3177.5. Its hypothesis, as its author stated it:

> A refinement of `nearest_decade_digit_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor. The one change is the metric for magnitude similarity: people treat the whole band of decades running from the lowest example's decade to the highest example's decade as familiar territory. So a target anywhere in that band, including decades between the examples that no example occupies, gets the full leading-digit boost, and the boost decays with each decade the target lies outside the band. This addresses the critique that people accept targets inside the examples' range more than targets outside it (range interpolation), while accepting adjacent-decade near-misses outside the band less than the nearest-decade model predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/decade_band_digit_common_factor.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.

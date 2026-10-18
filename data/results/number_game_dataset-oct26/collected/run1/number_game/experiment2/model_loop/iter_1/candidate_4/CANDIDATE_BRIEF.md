# Candidate Brief

Refine the incumbent: `nearest_decade_digit_common_factor` — rank 0, the best model on this data, ELPD-LOO -3193.5. Its hypothesis, as its author stated it:

> A refinement of `digit_similarity_with_common_factor`: people still say "yes" in proportion to the target's average final-digit similarity to the examples (same final digit most similar, same parity partly, opposite parity least), with an extra boost for multiples of the examples' common factor, but the one change is that they also compare the target's leading (tens) digit with the examples' and get a familiarity boost from the example whose decade is closest to the target's, a boost that is largest when the target shares that example's decade and decays with each decade further away. This addresses the critique that the model under-predicts "yes" for targets sharing a decade with an example (and, through that, for targets lying among the examples).

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment2/model_loop/models/nearest_decade_digit_common_factor.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.

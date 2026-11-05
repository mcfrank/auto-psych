# Candidate Brief

Refine the incumbent: `final_digit_overshadowing_overlap` — rank 0, the best model on this data, ELPD-LOO -3290.6. Its hypothesis, as its author stated it:

> Refinement of `one_sided_magnitude_bound_overlap` by one change: a final digit that is visibly shared by all of several examples (e.g. 28, 98) overshadows every other shared property. People still say "yes" in proportion to the salience-weighted share of the examples' shared profile that the target has, but when the examples visibly share a final digit, all the other shared properties (parity, multiples, contained digits, magnitude bounds, and so on) lose an inferred fraction of their salience. This disagrees most sharply with the incumbent on shared-final-digit sets: targets that keep the digit but break the other shared properties (28, 98 -> 18, 38) are endorsed far more, and targets that fit the other properties but not the digit (28, 98 -> 14, 56) far less.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/final_digit_overshadowing_overlap.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.

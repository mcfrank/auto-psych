# Candidate Brief

Refine the incumbent: `extended_multiples_concepts` — rank 0, the best model on this data, ELPD-LOO -1698.7. Its hypothesis, as its author stated it:

> People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples drawn from the concept without replacement), but their space of arithmetic concepts is larger: besides the familiar multiples of 3-12, they consider "multiples of m" for any m from 13 to 50 (multiples of 15, 20, 25, 45, ...), each such less-familiar multiples rule equally likely a priori and the family as a whole taking an inferred share of the rule prior. Because these rules are small, the size principle makes them dominate when every example is a multiple of a larger number (e.g. 45, 90 or 25, 75, 100), predicting fewer "yes" answers to targets that are only multiples of the smaller familiar factor (e.g. 27 or 60 after 90, 45).

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/extended_multiples_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.

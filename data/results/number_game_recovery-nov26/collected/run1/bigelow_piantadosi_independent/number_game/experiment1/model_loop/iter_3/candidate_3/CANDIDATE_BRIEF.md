# Candidate Brief

Refine the incumbent: `graded_profile_with_neighborhoods` — rank 0, the best model on this data, ELPD-LOO -802.4. Its hypothesis, as its author stated it:

> Refinement of `graded_feature_profile_matching`: people still judge membership by matching the target against a frequency-weighted profile of the examples' properties (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but the profile also contains each example's own numerical neighbourhood ("numbers close to 27") as one more graded property, with a fitted width, weighted by the same distinctiveness rule and shared in proportion to how many examples fall near it. The one change is this soft neighbourhood property: targets just one to three away from an example gain strong support that decays steeply within a few units, closing the critique that human generalisation falls off over short numerical distances far more sharply than the current models predict.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/graded_profile_with_neighborhoods.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.

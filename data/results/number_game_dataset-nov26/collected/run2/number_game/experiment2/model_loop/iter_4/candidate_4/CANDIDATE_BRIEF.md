# Candidate Brief

Refine the incumbent: `progression_band_first_digit_primacy` — rank 0, the best model on this data, ELPD-LOO -3160.0. Its hypothesis, as its author stated it:

> Refinement of `progression_graded_magnitude_band` (the incumbent) by one grafted component, taken from `graded_band_first_digit_primacy`: when several examples are shown without a shared final digit, the final digit of the FIRST example also lingers as a candidate pattern ("ends in 3"), with its own inferred salience, because the first example anchors people's initial guess before the later examples arrive. Everything else is the incumbent's judgment unchanged (salience-weighted share of the examples' shared profile, with the graded magnitude band, the "start at the smallest and add the gap" progression concept, multiples of the seed, final-digit overshadowing and the last example's lingering digit), so targets sharing the first-shown example's final digit are endorsed more than targets sharing no example's final digit, closing the first_example_digit_lift critique the incumbent fails.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/progression_band_first_digit_primacy.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
